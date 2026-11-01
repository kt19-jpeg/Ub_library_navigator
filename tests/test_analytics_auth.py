import pytest

from analytics.auth import (
    app_auth_configured,
    check_app_access,
    check_staff_access,
    staff_auth_configured,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("STAFF_DASHBOARD_AUTH_METHOD", raising=False)
    monkeypatch.delenv("STAFF_DASHBOARD_PASSWORD", raising=False)
    monkeypatch.delenv("APP_ACCESS_METHOD", raising=False)
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)


def test_correct_password_grants_access(monkeypatch):
    monkeypatch.setenv("STAFF_DASHBOARD_AUTH_METHOD", "password")
    monkeypatch.setenv("STAFF_DASHBOARD_PASSWORD", "secret123")
    assert check_staff_access("secret123") is True


def test_wrong_password_denies_access(monkeypatch):
    monkeypatch.setenv("STAFF_DASHBOARD_AUTH_METHOD", "password")
    monkeypatch.setenv("STAFF_DASHBOARD_PASSWORD", "secret123")
    assert check_staff_access("nope") is False


def test_unset_auth_method_denies_access(monkeypatch):
    # STAFF_DASHBOARD_AUTH_METHOD left unset entirely
    assert check_staff_access("anything") is False


def test_unrecognized_auth_method_denies_access(monkeypatch):
    monkeypatch.setenv("STAFF_DASHBOARD_AUTH_METHOD", "sso")
    assert check_staff_access("anything") is False


def test_password_method_without_password_set_denies_access(monkeypatch):
    monkeypatch.setenv("STAFF_DASHBOARD_AUTH_METHOD", "password")
    # STAFF_DASHBOARD_PASSWORD deliberately left unset
    assert check_staff_access("anything") is False


def test_staff_auth_configured_true_when_password_method_fully_set(monkeypatch):
    monkeypatch.setenv("STAFF_DASHBOARD_AUTH_METHOD", "password")
    monkeypatch.setenv("STAFF_DASHBOARD_PASSWORD", "secret123")
    assert staff_auth_configured() is True


def test_staff_auth_configured_false_when_unset(monkeypatch):
    assert staff_auth_configured() is False


def test_staff_auth_configured_false_when_password_missing(monkeypatch):
    monkeypatch.setenv("STAFF_DASHBOARD_AUTH_METHOD", "password")
    assert staff_auth_configured() is False


def test_app_access_is_independent_of_the_staff_gate(monkeypatch):
    # A correctly configured staff gate must not leak into app access, and
    # vice versa - they are separate credentials for separate audiences.
    monkeypatch.setenv("STAFF_DASHBOARD_AUTH_METHOD", "password")
    monkeypatch.setenv("STAFF_DASHBOARD_PASSWORD", "staff-secret")
    assert check_app_access("staff-secret") is False
    assert app_auth_configured() is False


def test_correct_app_password_grants_access(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_METHOD", "password")
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "app-secret")
    assert check_app_access("app-secret") is True
    assert check_staff_access("app-secret") is False


def test_wrong_app_password_denies_access(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_METHOD", "password")
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "app-secret")
    assert check_app_access("nope") is False


def test_app_auth_configured_true_when_password_method_fully_set(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_METHOD", "password")
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "app-secret")
    assert app_auth_configured() is True


def test_app_auth_configured_false_when_unset(monkeypatch):
    assert app_auth_configured() is False
