import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_llm(name, api_key, base_url, model):
    print(f"\n--- Testing {name} ---")
    if not api_key:
        print(f"ERROR: {name} API key not found in environment.")
        return
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"Sending request to {base_url} using {model}...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Return a JSON object with a 'status' field set to 'ok'."}
            ],
            response_format={"type": "json_object"}
        )
        
        print(f"Response received: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    # Test Moonshot
    test_llm(
        "Moonshot", 
        os.getenv("MOONSHOT_API_KEY"), 
        os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"), 
        os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")
    )
    
    # Test Mimo
    test_llm(
        "Mimo", 
        os.getenv("MIMO_API_KEY"), 
        os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"), 
        os.getenv("MIMO_MODEL", "mimo-v2.5-pro")
    )
