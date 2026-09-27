"""Telegram Bot API ile mesaj gönderimi. Token loglanmaz; tutarlar yalnızca yüzde.

Biçim: HTML (parse_mode="HTML"); dinamik metin html.escape ile kaçırılır. Telegram 400 (biçim hatası)
dönerse aynı mesaj düz metin olarak yeniden gönderilir — sabah mesajı asla biçim yüzünden düşmez.
"""

from __future__ import annotations

import html
import re

import requests
from loguru import logger

from janus.config import get_settings

API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_LEN = 4000
_TAGS = re.compile(r"</?(b|i|code|pre|u|s)>")


def _esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def _plain(text: str) -> str:
    return html.unescape(_TAGS.sub("", text))


def _post(token: str, chat_id: str, text: str, parse_mode: str | None, timeout: int) -> requests.Response:
    payload: dict[str, object] = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    return requests.post(API.format(token=token), json=payload, timeout=timeout)


def send_message(text: str, parse_mode: str | None = "HTML", timeout: int = 15) -> bool:
    s = get_settings()
    if not (s.telegram_bot_token and s.telegram_chat_id):
        logger.warning("Telegram yapılandırılmamış (.env: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        return False
    ok = True
    for i in range(0, len(text), MAX_LEN):
        chunk = text[i : i + MAX_LEN]
        try:
            r = _post(s.telegram_bot_token, s.telegram_chat_id, chunk, parse_mode, timeout)
            if r.status_code == 400 and parse_mode:  # biçim hatası → düz metin
                logger.warning("Telegram 400 (biçim); düz metinle yeniden deneniyor")
                r = _post(s.telegram_bot_token, s.telegram_chat_id, _plain(chunk), None, timeout)
            ok &= r.ok
            if not r.ok:
                logger.error("Telegram HTTP {} (token gizli)", r.status_code)
        except requests.RequestException as e:
            logger.error("Telegram gönderim hatası: {}", type(e).__name__)
            ok = False
    return ok


def format_quality(q: dict) -> str:
    """Veri sağlığı mesajı (HTML). Yalnızca sayılar ve kodlar."""
    if not q or not q.get("latest_date"):
        return "JANUS veri sağlığı: NAV verisi yok — ingest çalışmadı mı?"
    flag = "🟢" if q.get("ok") else "🔴"
    ref = q.get("reference_date")
    lines = [
        f"{flag} <b>JANUS veri sağlığı</b> — son NAV {_esc(q['latest_date'])}"
        + (f" (beklenen {_esc(ref)})" if ref else ""),
        f"Fon (NAV): {q['n_funds_nav']} | taze oran: %{q['fresh_ratio'] * 100:.1f} | gecikmeli: {q['n_stale']} | veri-eski: {q.get('n_data_stale', 0)}",
        f"Askıda (resmî): {q['n_suspended']} | alıma kapalı: {q['n_status_closed']} | bozuk hücre: {q['n_bad_cells']}",
        f"fund_master: {q['n_fund_master']} (info {q['n_info_ok']}, hist {q['n_hist_ok']})",
    ]
    if q.get("source_stale"):
        lines.append(
            f"<b>KAYNAK ESKİ</b>: son NAV beklenenden {q.get('fresh_days_behind', '?'):.0f} iş günü geride → emir üretilmez"
        )
    if q.get("macro_status"):
        lines.append(f"Makro: {_esc(q['macro_status'])}")
    if q.get("suspended_codes"):
        lines.append("Askıda: " + ", ".join(_esc(c) for c in q["suspended_codes"]))
    if not q.get("ok"):
        lines.append("<i>Evren yeterince taze değil → emir üretilmez (tut).</i>")
    return "\n".join(lines)


def format_orders(orders, absolute: bool = False) -> str:
    if orders is None or len(orders) == 0:
        return "Bugün emir yok (tut)."
    lines = []
    for leg, grp in orders.groupby("leg", sort=False):
        lines.append(f"<b>{_esc(str(leg).upper())}</b>")
        for row in grp.itertuples(index=False):  # rapor formatlama
            lines.append(
                f"• <code>{_esc(row.code)}</code> {_esc(row.action)} → hedef %{row.target_w * 100:.1f} "
                f"(Δ %{row.delta_w * 100:+.1f}) valör {_esc(row.valor)} — {_esc(row.reason)}"
            )
    return "\n".join(lines)
