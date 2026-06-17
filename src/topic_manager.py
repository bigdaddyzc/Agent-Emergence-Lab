"""Topic generation and switching helpers."""

from __future__ import annotations

from src.model_roles import get_helper_model


class TopicManager:
    def __init__(self, config: dict, client):
        self.config = config
        self.client = client

    def generate_next_topic(self, current_topic: str, retries: int = 3) -> str:
        model = get_helper_model(self.config)
        topic = self.client.generate_topic(model=model)
        attempts = 0
        while topic.lower() == current_topic.lower() and attempts < retries:
            topic = self.client.generate_topic(model=model)
            attempts += 1
        return topic

    def initial_topic(self) -> str:
        return self.client.generate_topic(model=get_helper_model(self.config))
