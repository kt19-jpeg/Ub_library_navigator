from evaluation.faithfulness import FaithfulnessVerdict, judge_faithfulness
from tests.fakes import FakeChatModel


def test_returns_true_for_a_grounded_verdict():
    model = FakeChatModel(FaithfulnessVerdict(grounded=True))
    result = judge_faithfulness(model, {"title": "Chemistry Basics", "subject": "Chemistry"}, "Covers chemistry.")
    assert result is True


def test_returns_false_for_an_ungrounded_verdict():
    model = FakeChatModel(FaithfulnessVerdict(grounded=False))
    result = judge_faithfulness(model, {"title": "X"}, "Proves a specific theorem not mentioned anywhere.")
    assert result is False


def test_prompt_includes_the_real_candidate_fields_and_reason():
    model = FakeChatModel(FaithfulnessVerdict(grounded=True))
    judge_faithfulness(model, {"title": "Chemistry Basics", "subject": "Chemistry"}, "Covers chemistry.")

    human = model.calls[0][1][1]
    assert "Chemistry Basics" in human
    assert "Covers chemistry." in human
    assert model.schemas == [FaithfulnessVerdict]


def test_dict_response_is_validated():
    model = FakeChatModel({"grounded": True})
    assert judge_faithfulness(model, {"title": "X"}, "reason") is True


def test_model_failure_fails_closed_as_ungrounded():
    model = FakeChatModel(RuntimeError("provider down"))
    assert judge_faithfulness(model, {"title": "X"}, "reason") is False


def test_unparseable_response_fails_closed():
    model = FakeChatModel(None)
    assert judge_faithfulness(model, {"title": "X"}, "reason") is False
