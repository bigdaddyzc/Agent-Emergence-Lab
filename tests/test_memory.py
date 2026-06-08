"""Tests for memory system: keyword extraction, similarity, retrieval."""

from src.memory import MemorySystem, MemoryEntry


def _make_ms():
    return MemorySystem({"memory": {}})


class TestKeywordExtraction:
    def test_english_keywords(self):
        ms = _make_ms()
        kw = ms._extract_keywords("terror management theory")
        assert "terror" in kw
        assert "management" in kw
        assert "theory" in kw
        # Stopwords should be excluded
        assert "the" not in kw
        assert "a" not in kw

    def test_chinese_keywords(self):
        ms = _make_ms()
        kw = ms._extract_keywords("人类喜欢看恐怖片是因为对未知的好奇心")
        assert "人类" in kw
        assert "恐怖片" in kw
        assert "好奇心" in kw
        # Noise should not appear
        assert "果重" not in kw  # common bigram artifact from old algorithm

    def test_chinese_noise_reduction(self):
        ms = _make_ms()
        kw = ms._extract_keywords("如果重力突然减半")
        assert "重力" in kw
        assert "突然" in kw
        assert "减半" in kw
        assert "如果" not in kw  # stopword
        # The old algorithm generated "果重", "力突", "然减" - these should NOT appear
        assert "果重" not in kw
        assert "力突" not in kw
        assert "然减" not in kw

    def test_mixed_language(self):
        ms = _make_ms()
        kw = ms._extract_keywords("AI 恐惧管理理论 terror management theory")
        assert "terror" in kw
        assert "management" in kw
        assert "theory" in kw
        assert "恐惧" in kw
        assert "管理" in kw
        assert "理论" in kw

    def test_empty_text(self):
        ms = _make_ms()
        kw = ms._extract_keywords("")
        assert kw == []

    def test_only_stopwords(self):
        ms = _make_ms()
        kw = ms._extract_keywords("的 了 在 是 a the")
        # Should not produce keywords from pure stopwords
        assert len(kw) == 0


class TestCosineSimilarity:
    def test_identical_vectors(self):
        ms = _make_ms()
        v = [1.0, 0.0, 0.0]
        assert ms._cosine_similarity(v, v) == 1.0

    def test_orthogonal_vectors(self):
        ms = _make_ms()
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(ms._cosine_similarity(a, b)) < 1e-6

    def test_partial_similarity(self):
        ms = _make_ms()
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = ms._cosine_similarity(a, b)
        assert 0.5 < sim < 1.0


class TestTextSimilarity:
    def test_identical_text(self):
        ms = _make_ms()
        assert ms._text_similarity(
            "人类喜欢看恐怖片",
            "人类喜欢看恐怖片",
            threshold=0.3,
        )

    def test_similar_text(self):
        ms = _make_ms()
        assert ms._text_similarity(
            "重力突然减半会怎样",
            "如果重力突然减半生活会变成什么样",
            threshold=0.3,
        )

    def test_different_text(self):
        ms = _make_ms()
        assert not ms._text_similarity(
            "人类喜欢看恐怖片",
            "外星人可能用什么方式娱乐自己",
            threshold=0.5,
        )


class TestShouldExtract:
    def test_turn_zero(self):
        ms = _make_ms()
        assert not ms.should_extract(0)

    def test_at_interval(self):
        ms = MemorySystem({"memory": {"extraction_interval": 3}})
        assert ms.should_extract(3)
        assert ms.should_extract(6)

    def test_not_at_interval(self):
        ms = MemorySystem({"memory": {"extraction_interval": 3}})
        assert not ms.should_extract(1)
        assert not ms.should_extract(4)


class TestRetrieveRelevant:
    def test_empty_long_term(self):
        ms = _make_ms()
        mems, info = ms.retrieve_relevant("anything")
        assert mems == []
        assert info["total_memories"] == 0

    def test_keyword_matching(self):
        ms = _make_ms()
        ms.long_term = [
            MemoryEntry(
                id="1", content="人类喜欢看恐怖片",
                source_agent="Nova", turn_number=1, timestamp=100,
                keywords=["人类", "喜欢", "恐怖片"],
            ),
            MemoryEntry(
                id="2", content="重力突然减半",
                source_agent="Riven", turn_number=2, timestamp=200,
                keywords=["重力", "突然", "减半"],
            ),
        ]
        mems, info = ms.retrieve_relevant("恐怖片", n=5)
        assert len(mems) >= 1
        assert mems[0].id == "1"
        assert "depth_distribution" in info

    def test_retrieval_count(self):
        ms = _make_ms()
        ms.long_term = [
            MemoryEntry(
                id=str(i), content=f"memory {i}",
                source_agent="Nova", turn_number=i, timestamp=i * 10,
                keywords=["memory"],
            )
            for i in range(5)
        ]
        mems, info = ms.retrieve_relevant("memory", n=3)
        assert len(mems) == 3


class TestClassifyDepth:
    def test_layer1_example(self):
        ms = _make_ms()
        assert ms.classify_depth("比如像自动驾驶这样的例子") == 1
        assert ms.classify_depth("有一次我看了一个恐怖片") == 1

    def test_layer2_pattern(self):
        ms = _make_ms()
        assert ms.classify_depth("人类总是会寻求刺激") == 2
        assert ms.classify_depth("这种情况往往出现在") == 2

    def test_layer3_principle(self):
        ms = _make_ms()
        assert ms.classify_depth("这背后的核心原理是") == 3
        assert ms.classify_depth("这个机制的逻辑是") == 3

    def test_layer4_application(self):
        ms = _make_ms()
        assert ms.classify_depth("这可以应用于实际场景") == 4

    def test_layer5_cross_domain(self):
        ms = _make_ms()
        assert ms.classify_depth("这可以联系到其他领域") == 5
        assert ms.classify_depth("两者在本质上是相通的") == 5

    def test_default_layer1(self):
        ms = _make_ms()
        assert ms.classify_depth("这是一个普通的陈述") == 1


class TestCompressContext:
    def test_no_compression_when_short(self):
        ms = MemorySystem({"memory": {"turns_kept_verbatim": 3}})
        ms.short_term = [
            {"agent": "Nova", "content": "hi", "token_count": 5, "turn_number": 0, "timestamp": 0},
        ]
        ms.compression_threshold = 0  # force should_compress=True
        assert not ms.should_compress()  # not enough compressible turns

    def test_compressed_summary_bounded(self, monkeypatch):
        ms = MemorySystem({"memory": {"turns_kept_verbatim": 2}})
        ms.short_term = [
            {"agent": "Nova", "content": "hello", "token_count": 1,
             "turn_number": i, "timestamp": i * 10}
            for i in range(4)
        ]
        # Mock summarize to return known values
        def fake_summarize(text):
            return "fake summary"

        monkeypatch.setattr(ms, '_summarize_turns', fake_summarize)

        # First compression
        ms.total_prompt_tokens = 9999
        assert ms.should_compress()
        ms.compress_context(3)
        first_len = len(ms.compressed_summary)
        assert "fake summary" in ms.compressed_summary

        # Second compression should NOT make the summary significantly longer
        ms.total_prompt_tokens = 9999
        ms.compress_context(6)
        second_len = len(ms.compressed_summary)
        assert second_len < first_len * 2 + 50  # not growing unboundedly
