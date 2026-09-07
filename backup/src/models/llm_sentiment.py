import os
from langchain_openai import ChatOpenAI
try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate

from src.utils.logger import log

class LLMSentiment:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            log.warning("OPENAI_API_KEY not found. LLM sentiment disabled.")
            self.llm = None
            return

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional crypto sentiment analyst. Analyze the following news/text and provide a sentiment score between -1 (extremely bearish) and 1 (extremely bullish). Return ONLY a JSON with 'score' and 'reasoning'."),
            ("user", "{text}")
        ])

    async def analyze_text(self, text):
        if not self.llm:
            return 0.0
        
        try:
            chain = self.prompt | self.llm
            response = await chain.ainvoke({"text": text})
            # Simplified parsing for demo
            import json
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            data = json.loads(content)
            log.info(f"LLM Sentiment: {data.get('score')} | {data.get('reasoning')}")
            return float(data.get("score", 0.0))
        except Exception as e:
            log.error(f"Error in LLM sentiment analysis: {e}")
            return 0.0
