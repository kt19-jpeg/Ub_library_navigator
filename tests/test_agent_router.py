import pytest

from agent.router import is_human_request


@pytest.mark.parametrize(
    "message",
    [
        "I want to talk to a human",
        "talk to a real person",
        "can I speak to a person please",
        "connect me with a librarian",
        "I'd like to talk to someone",
        "human",
        "Human.",
        "a real person",
        "can you connect me to a human?",
        "I need to speak with a librarian",
    ],
)
def test_detects_clear_human_requests(message):
    assert is_human_request(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "find me peer reviewed journals on mental health",
        "books about human resources management",
        "who is the subject librarian for chemistry",
        "articles about human rights law",
        "database for human anatomy",
        "I need help citing a source",
        "chemistry resources",
        "",
    ],
)
def test_does_not_trigger_on_ordinary_queries(message):
    assert is_human_request(message) is False


def test_case_insensitive():
    assert is_human_request("TALK TO A REAL PERSON") is True


def test_whitespace_only_bare_human_still_matches_with_surrounding_space():
    assert is_human_request("  human  ") is True
