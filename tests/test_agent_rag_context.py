from agent.models import RagContext
from pipeline.records import HELP_TOPIC, Record


def test_default_is_empty():
    assert RagContext().is_empty() is True


def test_not_empty_when_any_list_has_a_hit():
    ctx = RagContext(help_topics=[Record(id="help:a", type=HELP_TOPIC, text="A", metadata={})])
    assert ctx.is_empty() is False
