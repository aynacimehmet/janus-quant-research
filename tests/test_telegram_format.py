from janus.report.telegram import _plain, format_quality


def test_quality_html_escapes_and_no_markdown_underscore_problem():
    q = {
        "ok": True,
        "latest_date": "2026-09-21",
        "n_funds_nav": 2066,
        "fresh_ratio": 0.9797,
        "n_stale": 42,
        "n_suspended": 2,
        "suspended_codes": ["A<B", "C&D"],
        "n_bad_cells": 56,
        "n_status_closed": 1183,
        "n_fund_master": 2133,
        "n_info_ok": 2121,
        "n_hist_ok": 2066,
        "macro_status": "ok",
    }
    msg = format_quality(q)
    assert "<b>JANUS veri sağlığı</b>" in msg and "fund_master: 2133" in msg
    assert "A&lt;B" in msg and "C&amp;D" in msg and "<i>" not in msg
    plain = _plain(msg)
    assert "<b>" not in plain and "A<B" in plain and "C&D" in plain


def test_quality_not_ok_has_italic_note():
    q = {
        "ok": False,
        "latest_date": "2026-09-21",
        "n_funds_nav": 1,
        "fresh_ratio": 0.5,
        "n_stale": 1,
        "n_suspended": 0,
        "suspended_codes": [],
        "n_bad_cells": 0,
        "n_status_closed": 0,
        "n_fund_master": 1,
        "n_info_ok": 1,
        "n_hist_ok": 1,
    }
    assert "<i>Evren yeterince taze değil" in format_quality(q)
