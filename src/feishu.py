"""Feishu custom bot webhook client for sending agent messages to a group chat."""

import logging

import requests

logger = logging.getLogger(__name__)


class FeishuBot:
    """Sends messages to a Feishu group via a custom bot webhook."""

    def __init__(self, webhook_url: str, agent_name: str):
        self.webhook_url = webhook_url
        self.agent_name = agent_name

    def send_message(self, text: str) -> bool:
        """Send a plain text message. Returns True on success."""
        payload = {
            "msg_type": "text",
            "content": {
                "text": text,
            },
        }
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") != 0:
                logger.warning("Feishu API error (code=%s): %s",
                               body.get("code"), body.get("msg"))
                return False
            return True
        except requests.RequestException as e:
            logger.error("Feishu send failed for %s: %s", self.agent_name, e)
            return False
