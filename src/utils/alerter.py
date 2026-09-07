import os
from src.utils.logger import log

class TelegramAlerter:
    def __init__(self):
        self.enabled = False

    def send_message(self, message):
        # Disabled as per user request
        log.debug(f"Alert (Telegram Disabled): {message}")

alerter = TelegramAlerter()
