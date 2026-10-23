from agent.models import QueryPlan
from agent.planner import fallback_plan, plan_query
from tests.fakes import FakeChatModel


def test_returns_the_models_plan():
    plan = QueryPlan(keywords="mental health", resource_type="journals", peer_reviewed=True)
    model = FakeChatModel(plan)

    result = plan_query(model, "graduate", "find me peer reviewed journals on mental health")

    assert result == plan
    assert model.schemas == [QueryPlan]


def test_prompt_contains_role_message_and_recent_history():
    model = FakeChatModel(QueryPlan(keywords="x"))
    history = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(5)]

    plan_query(model, "faculty", "what about books?", history)

    system, human = model.calls[0]
    assert system[0] == "system" and human[0] == "human"
    text = human[1]
    assert "role: faculty" in text
    assert "message: what about books?" in text
    assert "user: q4" in text and "user: q2" in text
    assert "user: q1" not in text  # only the last 3 turns are sent


def test_dict_response_is_validated_into_a_plan():
    model = FakeChatModel({"keywords": "climate change", "resource_type": "books", "peer_reviewed": False})
    result = plan_query(model, "undergraduate", "books on climate change")
    assert result == QueryPlan(keywords="climate change", resource_type="books")


def test_invalid_dict_falls_back_to_raw_message():
    model = FakeChatModel({"keywords": "x", "resource_type": "podcasts"})
    assert plan_query(model, "graduate", "  some question ") == fallback_plan("some question")


def test_model_error_falls_back_to_raw_message():
    model = FakeChatModel(RuntimeError("provider down"))
    assert plan_query(model, "graduate", "some question") == QueryPlan(keywords="some question")


def test_none_response_falls_back():
    model = FakeChatModel(None)
    assert plan_query(model, "graduate", "some question") == QueryPlan(keywords="some question")


def test_empty_keywords_without_clarification_falls_back():
    model = FakeChatModel(QueryPlan(keywords="  "))
    assert plan_query(model, "graduate", "help") == QueryPlan(keywords="help")


def test_clarification_is_passed_through_even_with_empty_keywords():
    plan = QueryPlan(keywords="", needs_clarification="What topic are you researching?")
    model = FakeChatModel(plan)
    assert plan_query(model, "graduate", "find me something") == plan


def test_out_of_scope_note_is_passed_through_with_real_keywords():
    plan = QueryPlan(
        keywords="algorithms",
        out_of_scope_note="I can't prepare your midterm notes for you, but here are some resources on algorithms.",
    )
    model = FakeChatModel(plan)
    result = plan_query(
        model, "undergraduate", "can you help me prepare notes for my algorithms midterm from books available at UB"
    )
    assert result == plan


def test_wants_human_help_is_passed_through_with_the_topic_reused_from_history():
    plan = QueryPlan(keywords="mental health", wants_human_help=True)
    model = FakeChatModel(plan)
    history = [{"user": "find me peer reviewed journals on mental health", "assistant": "Here is what I found..."}]

    result = plan_query(
        model, "graduate", "these results are not what I need, can someone help me directly?", history
    )

    assert result == plan
