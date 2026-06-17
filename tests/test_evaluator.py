from src.evaluator import ConceptTracker, EmergenceSignalEvaluator


def test_evaluator_counts_core_signals():
    evaluator = EmergenceSignalEvaluator({
        "novelty_patterns": ["【新概念"],
        "analogy_patterns": ["类比"],
        "metacognitive_patterns": ["我不确定"],
        "reasoning_patterns": ["步骤1", "步骤2"],
        "knowledge_markers": ["关键发现"],
        "challenge_patterns": ["【质疑", "反例"],
    })
    text = "【新概念：镜像记忆】步骤1 先类比图书馆。步骤2 我不确定。【质疑：有反例吗】关键发现"
    signals = evaluator.evaluate_dict(text)
    assert signals["novel_concepts"] == 1
    assert signals["cross_domain_analogies"] == 1
    assert signals["metacognitive_count"] == 1
    assert signals["reasoning_steps"] == 2
    assert signals["critical_challenges"] >= 1
    assert signals["knowledge_generation"] == 1


def test_concept_tracker_records_named_concepts():
    tracker = ConceptTracker()
    events = tracker.observe(
        "这里有【新概念：记忆回声】可以继续发展。",
        turn_number=2,
        agent_name="Nova",
    )
    assert len(events) == 1
    assert events[0].name == "记忆回声"
    summary = tracker.summary()
    assert summary["记忆回声"]["mentions"] == 1
    assert summary["记忆回声"]["first_turn"] == 2
