from src.topic_manager import TopicManager


class TopicClient:
    def __init__(self):
        self.calls = 0

    def generate_topic(self, model=""):
        self.calls += 1
        return "旧话题" if self.calls == 1 else f"新话题-{model}"


def test_topic_manager_uses_helper_model_and_avoids_same_topic():
    client = TopicClient()
    manager = TopicManager({
        "agents": {"agent_a": {"model": "a"}, "agent_b": {"model": "b"}},
        "model_roles": {"helper_model": "h"},
    }, client)
    assert manager.generate_next_topic("旧话题") == "新话题-h"
    assert client.calls == 2
