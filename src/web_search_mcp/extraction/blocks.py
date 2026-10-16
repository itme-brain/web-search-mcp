"""Recognize access-check pages without rejecting articles discussing them."""

import re


def block_diagnostic(content: str | None, *, title: str | None = None) -> str | None:
    text = " ".join((content or "").lower().split())
    heading = (title or "").strip().lower()
    # Require paired challenge markers, rather than the word Cloudflare alone.
    if (
        "cloudflare ray id" in text
        and (heading in {"attention required! | cloudflare", "sorry, you have been blocked"}
             or text.lstrip("# ").startswith(("sorry, you have been blocked", "attention required", "please enable cookies", "you have been blocked")))
        and len(text.split()) < 1200
    ) or (
        (heading in {"just a moment...", "just a moment", "attention required! | cloudflare"} or text.startswith("just a moment"))
        and any(marker in text for marker in ("enable javascript and cookies", "verifying you are human", "checking your browser", "cloudflare"))
        and len(text.split()) < 400
    ):
        return "cloudflare_challenge"
    if len(text.split()) < 400 and re.search(r"\b(verify you are human|robot check|unusual traffic|complete the captcha)\b", text):
        return "captcha_or_bot_check"
    return None
