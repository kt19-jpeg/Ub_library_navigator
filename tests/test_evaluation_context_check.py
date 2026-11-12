from unittest.mock import MagicMock

from agent.graph import build_graph
from agent.models import FollowupAnswer, QueryPlan, Selection, SelectionItem
from evaluation.context_check import SCENARIO, Step, run_scenario
from onesearch.models import SearchCandidate, SearchResult
from tests.fakes import FakeChatModel


def _client():
    candidate = SearchCandidate(
        record_id="r1", title="A Book", author="Doe", resource_type="book", format=None,
        subject=None, description="About it.", date="2020", source="Alma",
        delivery_category="Alma-E", online_link=None, call_number=None, library=None,
        sub_location=None, availability_status=None,
    )
    client = MagicMock()
    client.search_with_fallback.return_value = SearchResult(
        candidates=[candidate], timed_out=False, fallback_link=None
    )
    client.fetch_detail.return_value = {}
    client.record_link.side_effect = lambda rid: f"https://rec/{rid}"
    client.fallback_link.side_effect = lambda q: f"https://more/{q}"
    return client


def _pick():
    return Selection(items=[SelectionItem(record_id="r1", reason="Fits.")], confidence="high")


def _scripted_model():
    # one scripted response per LLM call, in order: (plan[, selection | followup]) per turn
    return FakeChatModel(
        QueryPlan(keywords="cancer immunotherapy"), _pick(),
        QueryPlan(keywords="", chitchat="greeting"),
        QueryPlan(keywords="medicine", topic_category="library_services"), _pick(),
        QueryPlan(keywords="", chitchat="thanks"),
        QueryPlan(keywords="medicine", about_previous_results=True), FollowupAnswer(answer="They differ."),
        QueryPlan(keywords="diabetes"), _pick(),
        QueryPlan(keywords="diabetes", about_previous_results=True), FollowupAnswer(answer="Pick the first."),
    )


def test_the_seven_turn_scenario_routes_and_keeps_context():
    report = run_scenario(build_graph(_scripted_model(), _client()))
    assert report.ok, report.failures


def test_a_broken_router_is_actually_caught():
    # A model that treats the follow-up as a new search must fail the check, otherwise the
    # check proves nothing.
    model = FakeChatModel(
        QueryPlan(keywords="a"), _pick(),
        QueryPlan(keywords="", chitchat="greeting"),
        QueryPlan(keywords="m", topic_category="library_services"), _pick(),
        QueryPlan(keywords="", chitchat="thanks"),
        QueryPlan(keywords="m"), _pick(),  # follow-up planned as a plain search: wrong
        QueryPlan(keywords="d"), _pick(),
        QueryPlan(keywords="d", about_previous_results=True), FollowupAnswer(answer="x"),
    )
    report = run_scenario(build_graph(model, _client()))
    assert not report.ok
    assert any("turn 5" in f and "followup" in f for f in report.failures)


def test_scenario_covers_the_cases_that_broke_before():
    messages = [s.message for s in SCENARIO]
    assert len(messages) >= 7  # long enough that the 3-turn LLM history has been exceeded
    assert any(isinstance(s, Step) and "followup" in s.ran for s in SCENARIO)
