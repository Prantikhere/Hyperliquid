import os
import sys

# Ensure current directory is in path
sys.path.append(os.getcwd())

from src.execution.bingx_client import BingXClient
from src.data.multi_streamer import MultiExchangeStreamer
from src.agents.supervisor import SupervisorAgent
from src.execution.bingx_executor import BingXExecutor
from src.utils.logger import log

def verify():
    log.info("Verifying Trading BingX AI Agent System components...")
    
    try:
        log.info("1. Testing BingXClient instantiation...")
        client = BingXClient()
        
        log.info("2. Testing SupervisorAgent instantiation...")
        supervisor = SupervisorAgent()
        
        log.info("3. Testing BingXExecutor instantiation...")
        executor = BingXExecutor()

        log.info("4. Testing MultiExchangeStreamer instantiation...")
        streamer = MultiExchangeStreamer()
        
        log.info("SUCCESS: All core components instantiated correctly.")
        return True
    except Exception as e:
        log.error(f"FAILURE: Component instantiation failed: {e}")
        return False

if __name__ == "__main__":
    verify()
