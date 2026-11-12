from onesearch.session import SessionCookies, SessionManager

COOKIES_A = SessionCookies(jsessionid="A", urm_st="1000", urm_se="901000", institute="UB")
COOKIES_B = SessionCookies(jsessionid="B", urm_st="2000", urm_se="902000", institute="UB")


def test_mints_session_on_first_call():
    calls = []
    manager = SessionManager(mint_fn=lambda: calls.append(1) or COOKIES_A)
    cookies = manager.get_cookies()
    assert cookies == COOKIES_A
    assert len(calls) == 1


def test_reuses_session_within_window():
    calls = [COOKIES_A, COOKIES_B]
    manager = SessionManager(mint_fn=lambda: calls.pop(0))
    first = manager.get_cookies()
    second = manager.get_cookies()
    assert first == second == COOKIES_A


def test_remints_after_window_expires():
    calls = [COOKIES_A, COOKIES_B]
    manager = SessionManager(mint_fn=lambda: calls.pop(0), session_window_seconds=0)
    first = manager.get_cookies()
    second = manager.get_cookies()
    assert first == COOKIES_A
    assert second == COOKIES_B


def test_concurrent_callers_mint_only_once():
    import threading
    import time

    calls = []

    def slow_mint():
        calls.append(1)
        time.sleep(0.2)
        return COOKIES_A

    manager = SessionManager(mint_fn=slow_mint)
    threads = [threading.Thread(target=manager.get_cookies) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert len(calls) == 1


def test_warm_up_mints_in_the_background_so_the_first_call_is_instant():
    import time

    calls = []
    manager = SessionManager(mint_fn=lambda: calls.append(1) or COOKIES_A)
    manager.warm_up(poll_seconds=0.01)
    for _ in range(100):
        if calls:
            break
        time.sleep(0.02)
    assert manager.get_cookies() == COOKIES_A
    assert len(calls) == 1


def test_warm_up_refreshes_before_expiry_only_while_the_app_is_in_use():
    import time

    minted = []
    manager = SessionManager(
        mint_fn=lambda: (minted.append(1), COOKIES_A)[1], session_window_seconds=0.3,
    )
    manager.warm_up(refresh_margin_seconds=0.25, idle_limit_seconds=0.2, poll_seconds=0.01)
    time.sleep(0.15)
    manager.get_cookies()  # a "search" marks the app as in use
    time.sleep(0.5)
    refreshed_while_busy = len(minted)
    time.sleep(0.6)  # idle now: it must stop refreshing
    assert refreshed_while_busy >= 2
    assert len(minted) <= refreshed_while_busy + 1


def test_a_failing_background_mint_is_logged_not_raised(caplog):
    import logging
    import time

    def broken():
        raise RuntimeError("chromium crashed")

    manager = SessionManager(mint_fn=broken)
    with caplog.at_level(logging.WARNING):
        manager.warm_up(poll_seconds=0.01, retry_seconds=0.05)
        time.sleep(0.2)
    assert "chromium crashed" in caplog.text
