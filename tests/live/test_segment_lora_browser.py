"""Live Playwright test of SwarmUI's real Generate tab. No mock DOM.

Needs a running instance (SWARM_UI_URL) and a real listed adapter
(SWARM_UI_TEST_LORA). Does not start an image generation.
"""
from __future__ import annotations

import os
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright
import pytest


pytestmark = pytest.mark.inference


def test_real_browser_segment_selection_reaches_generation_parameter():
    base = os.environ.get("SWARM_UI_URL")
    lora_name = os.environ.get("SWARM_UI_TEST_LORA")
    if not base or not lora_name:
        raise RuntimeError(
            "LIVE BROWSER ACCEPTANCE NOT RUN: supply SWARM_UI_URL and "
            "SWARM_UI_TEST_LORA for the actual running SwarmUI and an installed "
            "adapter. A mocked Select element cannot prove browser behavior."
        )
    parsed = urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("SWARM_UI_URL must be a real HTTP(S) URL without credentials")
    state = os.environ.get("SWARM_UI_PLAYWRIGHT_STORAGE_STATE")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(**({"storage_state": state} if state else {}))
        try:
            page = context.new_page()
            page.goto(base, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_function(
                "typeof loraHelper !== 'undefined' && "
                "document.getElementById('input_prompt') !== null && "
                "document.getElementById('input_lorasectionconfinement') !== null",
                timeout=30000,
            )
            page.locator("#input_prompt").fill(
                "portrait <segment:face,1,0.8> apply this adapter to the face"
            )
            # Call the production selection handler on the real page, not a
            # synthetic test DOM. A model absent from the selector is a failure.
            page.evaluate("""name => {
                const input = document.getElementById('input_loras');
                if (!Array.from(input.options).some(o => o.value === name)) {
                    throw new Error('LoRA not available in real UI: ' + name);
                }
                if (loraHelper.selected.some(x => x.name === name)) {
                    loraHelper.selectLora(name);
                }
                loraHelper.selectLora(name);
            }""", lora_name)
            page.wait_for_function(
                "name => loraHelper.rendered[name] && "
                "Array.from(loraHelper.rendered[name].confinementInput.options)"
                ".some(x => x.value === '11')",
                arg=lora_name,
                timeout=15000,
            )
            page.evaluate("""name => {
                const select = loraHelper.rendered[name].confinementInput;
                select.value = '11';
                select.dispatchEvent(new Event('change', {bubbles: true}));
            }""", lora_name)
            assert page.locator("#input_lorasectionconfinement").input_value() == "11", (
                "Selected segment LoRA did not reach the real request parameter"
            )
            page.locator("#input_prompt").fill("portrait without a segment")
            selected = page.evaluate("""name => {
                const input = loraHelper.rendered[name].confinementInput;
                return {
                    value: input.value,
                    text: input.selectedOptions[0]?.text ?? ''
                };
            }""", lora_name)
            assert selected["value"] == "11", (
                "Editing the prompt silently turned a segment-confined adapter Global"
            )
            assert "not in current prompt" in selected["text"], selected
        finally:
            context.close()
            browser.close()
