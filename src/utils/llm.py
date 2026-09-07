import os
import httpx
import asyncio
from openai import AsyncOpenAI
from src.utils.logger import log

# POLICY: trade decisions are QUANT-ONLY and deterministic. The supervisor and all live
# executors decide from validated quant signals -- no LLM in any trading path. This client is
# retained only for non-trading uses (e.g. ad-hoc analysis) and is HARD-DISABLED for trading by
# default. To use it outside the trade loop, pass allow=True explicitly. This guard makes it
# structurally impossible to silently re-introduce LLM reliance into decisions.
class FallbackLLMClient:
    def __init__(self, allow: bool = False):
        if not allow and os.getenv("ALLOW_LLM", "no").lower() != "yes":
            raise RuntimeError(
                "FallbackLLMClient is disabled: trading decisions are quant-only. "
                "This client must NOT be used in any trade/decision path. "
                "For a non-trading use, construct with allow=True."
            )
        # Load API keys from environment
        self.mimo_key = os.getenv("MIMO_API_KEY")
        self.mimo_url = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
        self.mimo_model = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")
        
        self.weather_key = os.getenv("NVIDIA_WEATHER_KEY")
        self.nvidia_base_url = "https://integrate.api.nvidia.com/v1"
        
        # Reusable clients to prevent socket/file descriptor leaks
        self.mimo_client = AsyncOpenAI(api_key=self.mimo_key, base_url=self.mimo_url) if self.mimo_key else None
        self.weather_client = AsyncOpenAI(api_key=self.weather_key, base_url=self.nvidia_base_url) if self.weather_key else None
        
        # Define the fallback chain
        self.fallbacks = []
        self._initialize_fallbacks()

    def _initialize_fallbacks(self):
        # 1. Primary option: Mimo
        if self.mimo_key:
            self.fallbacks.append({
                "name": "Mimo",
                "client": self.mimo_client,
                "model": self.mimo_model,
                "temperature": 0.1,  # Keep it deterministic for quant/trading decisions
                "max_tokens": 4096,   # High limit to allow reasoning tokens
                "extra_body": {}
            })
            
        # 2. Nvidia Fallbacks
        nvidia_cfgs = [
            {
                "name": "Nvidia-Gemma",
                "model": "google/diffusiongemma-26b-a4b-it",
                "key": os.getenv("NVIDIA_FALLBACK_1_KEY"),
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 4096,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}
            },
            {
                "name": "Nvidia-MiniMax",
                "model": "minimaxai/minimax-m3",
                "key": os.getenv("NVIDIA_FALLBACK_2_KEY"),
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 8192,
                "extra_body": {}
            },
            {
                "name": "Nvidia-Nemotron",
                "model": "nvidia/nemotron-3-ultra-550b-a55b",
                "key": os.getenv("NVIDIA_FALLBACK_3_KEY"),
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 16384,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384}
            },
            {
                "name": "Nvidia-Kimi",
                "model": "moonshotai/kimi-k2.6",
                "key": os.getenv("NVIDIA_FALLBACK_4_KEY"),
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 16384,
                "extra_body": {}
            },
            {
                "name": "Nvidia-DeepSeek",
                "model": "deepseek-ai/deepseek-v4-pro",
                "key": os.getenv("NVIDIA_FALLBACK_5_KEY"),
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 16384,
                "extra_body": {"chat_template_kwargs": {"thinking": False}}
            },
            {
                "name": "Nvidia-GLM",
                "model": "z-ai/glm-5.1",
                "key": os.getenv("NVIDIA_FALLBACK_6_KEY"),
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 16384,
                "extra_body": {}
            }
        ]
        
        for cfg in nvidia_cfgs:
            if cfg["key"]:
                self.fallbacks.append({
                    "name": cfg["name"],
                    "client": AsyncOpenAI(api_key=cfg["key"], base_url=self.nvidia_base_url),
                    "model": cfg["model"],
                    "temperature": cfg["temperature"],
                    "top_p": cfg["top_p"],
                    "max_tokens": cfg["max_tokens"],
                    "extra_body": cfg["extra_body"]
                })

        # 3. Claude fallback (Direct Anthropic HTTP call)
        self.claude_key = os.getenv("CLAUDE_API_KEY")
        
        # 4. Gemini fallback (Direct Gemini HTTP call)
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    async def chat_completion(self, messages, is_weather=False, temperature=0.1, max_tokens=1024):
        # If is_weather is requested, try Nvidia Weather predictor first
        if is_weather and self.weather_key:
            try:
                response = await self.weather_client.chat.completions.create(
                    model="nvidia/nemotron-3-ultra-550b-a55b", 
                    messages=messages,
                    temperature=1.0,
                    top_p=0.95,
                    max_tokens=16384,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384},
                    timeout=60.0
                )
                content = response.choices[0].message.content
                if content:
                    log.info("LLM Call successful using dedicated Weather model.")
                    return content
            except Exception as e:
                log.error(f"Dedicated Weather LLM failed: {e}. Falling back to standard chain...")

        # === PHASE 1: GEMINI (Direct & Mimo) ===
        
        # 1.1 Try Gemini direct via HTTP if available
        if self.gemini_key:
            try:
                log.info("Attempting LLM call using Gemini direct HTTP...")
                async with httpx.AsyncClient() as client:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
                    headers = {"Content-Type": "application/json"}
                    
                    contents = []
                    for m in messages:
                        role = "model" if m["role"] == "assistant" else "user"
                        contents.append({"role": role, "parts": [{"text": m["content"]}]})
                        
                    payload = {
                        "contents": contents,
                        "generationConfig": {
                            "temperature": temperature,
                            "maxOutputTokens": max_tokens
                        }
                    }
                    response = await client.post(url, headers=headers, json=payload, timeout=60.0)
                    response.raise_for_status()
                    res_json = response.json()
                    content = res_json["candidates"][0]["content"]["parts"][0]["text"]
                    if content:
                        log.info("LLM Call successful using Gemini direct API.")
                        return content
            except Exception as e:
                log.error(f"Gemini direct HTTP call failed: {e}")

        # 1.2 Try Mimo if available
        if self.mimo_key:
            try:
                response = await self.mimo_client.chat.completions.create(
                    model=self.mimo_model,
                    messages=messages,
                    temperature=0.1,  # Keep it deterministic for quant/trading decisions
                    max_tokens=4096,   # High limit to allow reasoning tokens
                    timeout=60.0
                )
                content = response.choices[0].message.content
                if content:
                    log.info("LLM Call successful using Mimo.")
                    return content
            except Exception as e:
                log.error(f"Mimo LLM call failed: {e}")

        # === PHASE 2: NVIDIA FALLBACKS ===
        for fb in self.fallbacks:
            if fb["name"] == "Mimo":
                continue  # Skip Mimo here, as it was already tried under the Gemini Phase
            try:
                log.info(f"Attempting LLM call using {fb['name']}...")
                
                # Determine parameters for this call
                req_temp = fb.get("temperature") if fb.get("temperature") is not None else temperature
                req_max_tokens = fb.get("max_tokens") if fb.get("max_tokens") is not None else max_tokens
                req_top_p = fb.get("top_p")
                
                kwargs = {
                    "model": fb["model"],
                    "messages": messages,
                    "temperature": req_temp,
                    "max_tokens": req_max_tokens,
                    "timeout": 60.0
                }
                if req_top_p is not None:
                    kwargs["top_p"] = req_top_p
                if fb["extra_body"]:
                    kwargs["extra_body"] = fb["extra_body"]
                
                response = await fb["client"].chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content:
                    log.info(f"LLM Call successful using {fb['name']}.")
                    return content
            except Exception as e:
                log.error(f"LLM {fb['name']} call failed: {e}")

        # === PHASE 3: CLAUDE ===
        if self.claude_key:
            try:
                log.info("Attempting LLM call using Claude direct HTTP...")
                async with httpx.AsyncClient() as client:
                    headers = {
                        "x-api-key": self.claude_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    }
                    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
                    user_messages = [m for m in messages if m["role"] != "system"]
                    payload = {
                        "model": "claude-sonnet-4-6",
                        "max_tokens": max_tokens,
                        "messages": [{"role": m["role"], "content": m["content"]} for m in user_messages],
                        "system": system_prompt,
                        "temperature": temperature
                    }
                    response = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=60.0)
                    response.raise_for_status()
                    res_json = response.json()
                    content = res_json["content"][0]["text"]
                    if content:
                        log.info("LLM Call successful using Claude.")
                        return content
            except Exception as e:
                log.error(f"Claude direct HTTP call failed: {e}")

        log.critical("ALL LLM fallback options exhausted. Failsafe activated.")
        return None
