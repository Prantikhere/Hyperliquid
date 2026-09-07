import os
import json
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from src.utils.logger import log
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()

class VectorMemory:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = "trading-memory"
        self.dimension = 384  # Dimension for 'all-MiniLM-L6-v2'
        
        if not self.api_key:
            log.warning("PINECONE_API_KEY not found in environment. Vector memory disabled.")
            self.pc = None
            return

        try:
            self.pc = Pinecone(api_key=self.api_key)
            
            # Initialize sentence transformer for embeddings
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create index if it doesn't exist
            active_indexes = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in active_indexes:
                log.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            
            self.index = self.pc.Index(self.index_name)
            log.info("Pinecone Vector Memory initialized successfully.")
        except Exception as e:
            log.error(f"Failed to initialize Pinecone: {e}")
            self.pc = None

    def store_decision(self, symbol, context, decision):
        """Store a trading decision and its context as a vector."""
        if not self.pc: return
        
        try:
            # Create a string representation of the market context for embedding
            context_str = f"Symbol: {symbol}, Price: {context['current_price']}, RSI: {context['indicators'].get('rsi_14')}, Trend: {context['indicators'].get('trend')}, Regime: {context.get('market_regime')}"
            embedding = self.model.encode(context_str).tolist()
            
            metadata = {
                "symbol": symbol,
                "action": decision.get("action", "HOLD"),
                "confidence": float(decision.get("confidence", 0)),
                "reason": decision.get("reason", ""),
                "price": float(context["current_price"]),
                "regime": context.get("market_regime", "UNKNOWN"),
                "timestamp": int(np.datetime64('now').astype(int))
            }
            
            vector_id = f"{symbol}_{int(np.random.rand()*1000000)}" 
            self.index.upsert(vectors=[(vector_id, embedding, metadata)])
            log.debug(f"Stored decision memory for {symbol}")
        except Exception as e:
            log.error(f"Pinecone store error: {e}")

    def retrieve_similar_decisions(self, symbol, context, top_k=3):
        """Retrieve similar past decisions based on current context."""
        if not self.pc: return []
        
        try:
            context_str = f"Symbol: {symbol}, Price: {context['current_price']}, RSI: {context['indicators'].get('rsi_14')}, Trend: {context['indicators'].get('trend')}, Regime: {context.get('market_regime')}"
            embedding = self.model.encode(context_str).tolist()
            
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"symbol": {"$eq": symbol}}
            )
            
            return [res['metadata'] for res in results['matches']]
        except Exception as e:
            log.error(f"Pinecone retrieval error: {e}")
            return []
