"""Tests for agent: build_context, clean_response, extract_references."""

from src.agent import Agent, AgentConfig


def _make_agent(name="Nova", partner="Riven", temp=0.75):
    cfg = AgentConfig(
        name=name, model="test-model", role="",
        system_prompt_template="{partner_name} 和 {topic}",
        temperature=temp, max_tokens=256, context_window=4096,
    )
    return Agent(cfg, partner, None)


class TestCleanResponse:
    def test_no_prefix(self):
        agent = _make_agent()
        text = "我觉得人类挺奇怪的"
        assert agent.clean_response(text) == text

    def test_strips_own_name_prefix(self):
        agent = _make_agent()
        text = "Nova：我觉得人类挺奇怪的"
        cleaned = agent.clean_response(text)
        assert cleaned == "我觉得人类挺奇怪的"

    def test_strips_own_name_colon(self):
        agent = _make_agent()
        text = "Nova: hello"
        assert agent.clean_response(text) == "hello"

    def test_strips_partner_lines_self_dialogue(self):
        agent = _make_agent()
        text = "我觉得人类挺奇怪的\nRiven：你说得对！\n还有就是"
        cleaned = agent.clean_response(text)
        # "Riven：你说得对！" should be removed (self-dialogue)
        assert "Riven" not in cleaned
        assert "我觉得人类挺奇怪的" in cleaned
        assert "还有就是" in cleaned

    def test_preserves_partner_name_in_content(self):
        """Partner name mentioned in dialogue should be preserved."""
        agent = _make_agent()
        text = "你说得对，我觉得Riven说得有道理"
        assert agent.clean_response(text) == text

    def test_empty_text(self):
        agent = _make_agent()
        assert agent.clean_response("") == ""

    def test_only_prefix(self):
        agent = _make_agent()
        assert agent.clean_response("Nova：") == ""


class TestExtractReferences:
    def test_count_zero(self):
        agent = _make_agent()
        assert agent.extract_references("完全无关的内容") == 0

    def test_count_partner_name(self):
        agent = _make_agent()
        assert agent.extract_references("Riven 你说得对") >= 1

    def test_count_reference_patterns(self):
        agent = _make_agent()
        text = "正如你说的，Riven的想法很有启发性"
        assert agent.extract_references(text) >= 1

    def test_multiple_references(self):
        agent = _make_agent()
        text = "Riven你刚才说的观点让我想到另一个角度。基于你的想法，我觉得..."
        assert agent.extract_references(text) >= 2

    def test_empty_string(self):
        agent = _make_agent()
        assert agent.extract_references("") == 0


class TestBuildContext:
    def test_empty_history(self):
        agent = _make_agent()
        ctx = agent.build_context(conversation_history=[])
        assert ctx is not None
        assert len(ctx) > 0
        # Should mention starting the conversation
        assert "随便聊" in ctx

    def test_with_history(self):
        agent = _make_agent()
        history = [
            {"agent": "Nova", "content": "你好", "token_count": 2, "turn_number": 0, "timestamp": 0},
            {"agent": "Riven", "content": "你好呀", "token_count": 3, "turn_number": 0, "timestamp": 1},
        ]
        ctx = agent.build_context(conversation_history=history)
        assert "你好" in ctx
        assert "你好呀" in ctx
        # Should not have "Nova：你好" format (causes self-dialogue)
        assert "Nova：你好" not in ctx

    def test_with_memories(self):
        agent = _make_agent()
        from src.memory import MemoryEntry
        mem = MemoryEntry(
            id="1", content="人类喜欢看恐怖片",
            source_agent="Nova", turn_number=1, timestamp=0,
        )
        ctx = agent.build_context(conversation_history=[], memories=[mem])
        assert "人类喜欢看恐怖片" in ctx
        # New memory header text
        assert "共同积累的见解" in ctx

    def test_with_compressed_summary(self):
        agent = _make_agent()
        history = [{"agent": "Nova", "content": str(i), "token_count": 1,
                     "turn_number": i, "timestamp": i * 10} for i in range(5)]
        ctx = agent.build_context(
            conversation_history=history,
            compressed_summary="前面聊了恐怖片",
        )
        assert "前面聊了恐怖片" in ctx

    def test_with_depth_label(self):
        agent = _make_agent()
        ctx = agent.build_context(
            conversation_history=[],
            depth_level=2,
            depth_label="模式与连接",
        )
        assert "第2层" in ctx
        assert "模式与连接" in ctx

    def test_build_instruction_depth1(self):
        agent = _make_agent()
        inst = agent._build_instruction(1, 1, 0)
        assert "对方" in inst or "你" in inst

    def test_build_instruction_depth3(self):
        agent = _make_agent()
        inst = agent._build_instruction(3, 1, 0)
        assert "总结" in inst  # depth >= 3 adds synthesis step

    def test_build_instruction_low_refs(self):
        agent = _make_agent()
        inst = agent._build_instruction(1, 0, 2)
        assert "多回应" in inst


class TestCompileSystemPrompt:
    def test_template_rendering(self):
        cfg = AgentConfig(
            name="Nova", model="test", role="",
            system_prompt_template="你好{partner_name}，我们聊聊{topic}",
            temperature=0.5, max_tokens=100, context_window=512,
        )
        agent = Agent(cfg, "Riven", None)
        prompt = agent.compile_system_prompt("恐怖片")
        assert "Riven" in prompt
        assert "恐怖片" in prompt
