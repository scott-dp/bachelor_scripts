#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
from pathlib import Path
from datetime import datetime

URL = "https://bike.shimano.com/en-NA/products/apps/e-tube-project-professional.html"

MODELS = [
    "SC-E6010",
    "DU-E8000",
    "DU-E6100",
    "DU-E6002",
    "DU-E6001",
    "DU-E5000",
]

OUT_DIR = Path("shimano_firmware_changelogs")


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def parse_rows_from_html(html: str, models: list[str]) -> dict[str, list[dict]]:
    """
    Returns:
      { model: [ {date, version, message}, ... ] }
    """
    from bs4 import BeautifulSoup  # pip install beautifulsoup4

    soup = BeautifulSoup(html, "html.parser")

    # Table wrapper div (as you pointed out)
    table_div = soup.select_one("div.firmware-table-details")
    if not table_div:
        # If this is missing, page likely needs JS rendering
        return {m: [] for m in models}

    results: dict[str, list[dict]] = {m: [] for m in models}

    for model in models:
        # Each entry row: <tr data-modelno="DU-E5000"> ... </tr>
        rows = table_div.select(f'tr[data-modelno="{model}"]')
        for tr in rows:
            date_el = tr.select_one(".firmware-date")
            version_el = tr.select_one(".firmware-version")
            msg_el = tr.select_one(".firmware-message")

            entry = {
                "date": normalize_space(date_el.get_text()) if date_el else "",
                "version": normalize_space(version_el.get_text()) if version_el else "",
                "message": normalize_space(msg_el.get_text()) if msg_el else "",
            }

            # Only keep if we got a message (you said this is the key field)
            if entry["message"]:
                results[model].append(entry)

    return results


def fetch_with_requests(url: str) -> str:
    import requests  # pip install requests

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text

def fetch_with_playwright(url: str) -> str:
    """
    Robust fetch with debugging artifacts.
    Creates:
      - debug_shimano.png
      - debug_shimano.html
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,  # IMPORTANT: easier to see consent/blocks
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport={"width": 1400, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=120000)

        # Try to dismiss common cookie/consent dialogs if present
        # (These selectors are generic; harmless if they don't exist)
        for sel in [
            "button:has-text('Accept')",
            "button:has-text('I Agree')",
            "button:has-text('Agree')",
            "button:has-text('OK')",
            "button:has-text('Got it')",
        ]:
            try:
                page.locator(sel).first.click(timeout=1500)
                break
            except Exception:
                pass

        # Scroll to trigger lazy-loading
        page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.6)")
        page.wait_for_timeout(1500)

        # Wait for the title marker you mentioned (more stable)
        # <p class="firmware-title">FIRMWARE UPDATES</p>
        try:
            page.wait_for_selector("p.firmware-title", timeout=60000)
        except Exception:
            # Save debug output and raise
            page.screenshot(path="debug_shimano.png", full_page=True)
            Path("debug_shimano.html").write_text(page.content(), encoding="utf-8")
            browser.close()
            raise

        # Now wait for the table container *to exist* (not necessarily visible)
        try:
            page.wait_for_selector("div.firmware-table-details", state="attached", timeout=60000)
        except Exception:
            page.screenshot(path="debug_shimano.png", full_page=True)
            Path("debug_shimano.html").write_text(page.content(), encoding="utf-8")
            browser.close()
            raise

        html = page.content()
        browser.close()
        return html


def write_model_files(data: dict[str, list[dict]]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for model, entries in data.items():
        out_path = OUT_DIR / f"{safe_filename(model)}.txt"

        # Sort by date if it’s parseable; otherwise keep original order.
        def date_key(e):
            # Shimano shows dates like "Aug 19, 2025"
            # If parsing fails, sort key becomes empty and stable sort keeps order.
            try:
                return datetime.strptime(e["date"], "%b %d, %Y")
            except Exception:
                return datetime.min

        entries_sorted = sorted(entries, key=date_key, reverse=True)

        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"Model: {model}\n")
            f.write(f"Source: {URL}\n")
            f.write(f"Fetched: {timestamp}\n")
            f.write("=" * 80 + "\n\n")

            if not entries_sorted:
                f.write("No entries found.\n")
                continue

            for e in entries_sorted:
                # Include context (date/version) + the message you need
                date_str = e["date"] or "Unknown date"
                ver_str = e["version"] or "Unknown version"
                msg = e["message"]

                f.write(f"- {date_str} | {ver_str}\n")
                f.write(f"  {msg}\n\n")


def main():
    # 1) Try requests first
    try:
        html = fetch_with_requests(URL)
        data = parse_rows_from_html(html, MODELS)

        # If we got at least one message anywhere, we're good.
        if any(data[m] for m in MODELS):
            write_model_files(data)
            print(f"Done (requests). Files written to: {OUT_DIR.resolve()}")
            return

        print("Requests fetch did not find firmware table rows; falling back to Playwright...")
    except Exception as e:
        print(f"Requests failed ({e}); falling back to Playwright...")

    # 2) Fallback to Playwright (JS-rendered)
    try:
        html = fetch_with_playwright(URL)
        data = parse_rows_from_html(html, MODELS)
        write_model_files(data)
        print(f"Done (playwright). Files written to: {OUT_DIR.resolve()}")
    except Exception as e:
        print("Playwright failed too.")
        print("Error:", e)
        print("\nTry installing deps:")
        print("  pip install requests beautifulsoup4 playwright")
        print("  playwright install")
        sys.exit(1)


if __name__ == "__main__":
    main()
