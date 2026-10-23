from typing import get_args

import pytest
from pydantic import ValidationError

from agent.models import ROLES, QueryPlan, ResourceType, Selection
from onesearch.client import RESOURCE_TYPES


def test_resource_type_literal_matches_layer1_resource_types():
    assert set(get_args(ResourceType)) == set(RESOURCE_TYPES)


def test_query_plan_defaults():
    plan = QueryPlan(keywords="mental health")
    assert plan.resource_type is None
    assert plan.peer_reviewed is False
    assert plan.needs_clarification is None
    assert plan.out_of_scope_note is None


def test_query_plan_rejects_unknown_resource_type():
    with pytest.raises(ValidationError):
        QueryPlan(keywords="x", resource_type="podcasts")


def test_selection_requires_valid_confidence():
    with pytest.raises(ValidationError):
        Selection(items=[], confidence="certain")


def test_roles_are_the_three_project_roles():
    assert ROLES == ("undergraduate", "graduate", "faculty")
