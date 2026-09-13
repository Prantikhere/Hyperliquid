"""
RNN-based price prediction model for trade signal enhancement.
Uses LSTM/GRU to learn temporal patterns from price sequences and predict short-term direction.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from src.utils.logger import log


class PricePredictorRNN(nn.Module):
    """
    LSTM-based model for predicting short-term price direction.
    
    Architecture:
    - Input: Sequence of price features (OHLCV + technical indicators)
    - Hidden: 2 LSTM layers with 64 hidden units
    - Output: Probability of price going UP (0-1)
    """
    
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, dropout=0.2):
        super(PricePredictorRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism for focusing on important time steps
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output prediction
        output = self.fc(context)
        return output


class RNNTradePredictor:
    """
    Complete RNN-based trade prediction system.
    Handles data preparation, training, and inference.
    """
    
    def __init__(self, model_path="models_local/rnn_price_predictor.pth"):
        self.model_path = model_path
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.sequence_length = 20  # Look back 20 periods
        self.min_confidence = 0.55  # Minimum confidence to trade
        
        # Load model if exists
        self._load_model()
    
    def _prepare_features(self, prices):
        """
        Prepare features from price data.
        Features: OHLCV-like + technical indicators
        """
        if len(prices) < 30:
            return None
        
        try:
            prices = np.array(prices, dtype=float)
            
            # Basic price features
            returns = np.diff(prices) / prices[:-1]
            log_returns = np.log(prices[1:] / prices[:-1])
            
            # Volatility features
            volatility_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
            volatility_10 = np.std(returns[-10:]) if len(returns) >= 10 else 0
            volatility_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0
            
            # Momentum features
            momentum_5 = (prices[-1] / prices[-6] - 1) if len(prices) >= 6 else 0
            momentum_10 = (prices[-1] / prices[-11] - 1) if len(prices) >= 11 else 0
            momentum_20 = (prices[-1] / prices[-21] - 1) if len(prices) >= 21 else 0
            
            # Mean reversion features
            ma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
            ma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
            ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            
            # Distance from moving averages
            dist_ma5 = (prices[-1] - ma_5) / ma_5
            dist_ma10 = (prices[-1] - ma_10) / ma_10
            dist_ma20 = (prices[-1] - ma_20) / ma_20
            
            # RSI-like feature
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi_normalized = rsi / 100
            
            # Combine features
            features = np.array([
                volatility_5,
                volatility_10,
                volatility_20,
                momentum_5,
                momentum_10,
                momentum_20,
                dist_ma5,
                dist_ma10,
                dist_ma20,
                rsi_normalized,
                returns[-1] if len(returns) > 0 else 0,
                log_returns[-1] if len(log_returns) > 0 else 0
            ])
            
            return features
            
        except Exception as e:
            log.error(f"Error preparing features: {e}")
            return None
    
    def _create_sequences(self, feature_matrix):
        """Create sequences for LSTM input."""
        sequences = []
        for i in range(len(feature_matrix) - self.sequence_length + 1):
            seq = feature_matrix[i:i + self.sequence_length]
            sequences.append(seq)
        return np.array(sequences)
    
    def train(self, price_data, epochs=50, learning_rate=0.001):
        """
        Train the RNN model on historical price data.
        
        Args:
            price_data: List of historical prices
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        if len(price_data) < 100:
            log.warning("Not enough data for RNN training (need 100+ prices)")
            return False
        
        try:
            # Prepare features
            features = []
            for i in range(30, len(price_data)):
                feat = self._prepare_features(price_data[:i+1])
                if feat is not None:
                    features.append(feat)
            
            if len(features) < self.sequence_length + 10:
                log.warning("Not enough sequences for training")
                return False
            
            features = np.array(features)
            
            # Normalize features
            self.scaler_mean = np.mean(features, axis=0)
            self.scaler_std = np.std(features, axis=0) + 1e-10
            features_normalized = (features - self.scaler_mean) / self.scaler_std
            
            # Create sequences
            X = self._create_sequences(features_normalized)
            
            # Create labels (1 if price went up, 0 if down)
            y = []
            for i in range(self.sequence_length, len(features)):
                # Next period return
                next_return = (price_data[i+1] - price_data[i]) / price_data[i] if i+1 < len(price_data) else 0
                y.append(1 if next_return > 0 else 0)
            
            y = np.array(y[:len(X)])
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            
            # Initialize model
            input_size = X.shape[2]
            self.model = PricePredictorRNN(input_size=input_size)
            
            # Loss and optimizer
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    accuracy = ((outputs > 0.5).float() == y_tensor).float().mean()
                    log.info(f"RNN Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
            
            # Save model
            self._save_model()
            
            log.info(f"RNN training complete. Final accuracy: {accuracy.item():.4f}")
            return True
            
        except Exception as e:
            log.error(f"RNN training error: {e}")
            return False
    
    def predict(self, prices):
        """
        Predict price direction.
        
        Args:
            prices: Recent price history
            
        Returns:
            dict with 'prediction' (0-1), 'confidence', 'signal' (LONG/SHORT/NEUTRAL)
        """
        if self.model is None:
            return {"prediction": 0.5, "confidence": 0, "signal": "NEUTRAL"}
        
        try:
            # Prepare features
            features = self._prepare_features(prices)
            if features is None:
                return {"prediction": 0.5, "confidence": 0, "signal": "NEUTRAL"}
            
            # Normalize
            if self.scaler_mean is not None and self.scaler_std is not None:
                features_normalized = (features - self.scaler_mean) / self.scaler_std
            else:
                return {"prediction": 0.5, "confidence": 0, "signal": "NEUTRAL"}
            
            # Create sequence (use last sequence_length points)
            if len(features_normalized) >= self.sequence_length:
                seq = features_normalized[-self.sequence_length:]
            else:
                seq = features_normalized
            
            # Reshape for model
            X = torch.FloatTensor(seq).unsqueeze(0)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(X).item()
            
            # Calculate confidence
            confidence = abs(prediction - 0.5) * 2  # 0-1 scale
            
            # Determine signal
            if prediction > 0.55:
                signal = "LONG"
            elif prediction < 0.45:
                signal = "SHORT"
            else:
                signal = "NEUTRAL"
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "signal": signal
            }
            
        except Exception as e:
            log.error(f"RNN prediction error: {e}")
            return {"prediction": 0.5, "confidence": 0, "signal": "NEUTRAL"}
    
    def _save_model(self):
        """Save model to disk."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler_mean': self.scaler_mean,
                'scaler_std': self.scaler_std,
                'sequence_length': self.sequence_length
            }, self.model_path)
            log.info(f"RNN model saved to {self.model_path}")
        except Exception as e:
            log.error(f"Error saving RNN model: {e}")
    
    def _load_model(self):
        """Load model from disk."""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path)
                self.model = PricePredictorRNN()
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.scaler_mean = checkpoint['scaler_mean']
                self.scaler_std = checkpoint['scaler_std']
                self.sequence_length = checkpoint['sequence_length']
                log.info(f"RNN model loaded from {self.model_path}")
                return True
        except Exception as e:
            log.error(f"Error loading RNN model: {e}")
        return False


# Global instance
rnn_predictor = RNNTradePredictor()
