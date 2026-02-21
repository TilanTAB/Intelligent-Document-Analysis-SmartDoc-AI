from intelligence.orchestrator import AgentState


def test_agent_state_includes_retry_feedback_fields():
    annotations = getattr(AgentState, "__annotations__", {})

    assert "should_retry" in annotations
    assert "feedback_for_research" in annotations
