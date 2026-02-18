#!/usr/bin/env python3
"""
Tölzer Kurier / Merkur - Täglicher PDF-Download
Playwright: Login + API-Request-Interception für Issue-ID + Download

Installation:
  pip install playwright
  playwright install chromium

Cron (täglich 06:30):
  30 6 * * * /home/gh/python/venv_py311/bin/python3 /home/gh/python/merkur_download.py
"""

import re
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ── Konfiguration ──────────────────────────────────────────────────────────────
USER         = 'gh@heissa.de'
PASSWD       = '2026Einfach!'
PUBLICATION  = 'toelzer-kurier-tk'
BASE_URL     = 'https://abo.merkur.de'
LOGIN_URL    = 'https://abo.merkur.de/anmelden'
DOWNLOAD_DIR = Path('/home/gh/python/zeitung')
DEBUG_DIR    = Path('/home/gh/python/zeitung/debug')

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def dismiss_cookie_banner(page):
    """Schließt CookieYes-Banner falls vorhanden."""
    for sel in ['button:has-text("Alle akzeptieren")', 'button:has-text("Akzeptieren")',
                '.cky-btn-accept', '#ckySaveBtn']:
        loc = page.locator(sel)
        if loc.count() > 0:
            try:
                loc.first.click(timeout=3_000)
                page.wait_for_selector('.cky-overlay', state='hidden', timeout=5_000)
                logger.info('Cookie-Banner geschlossen')
                return
            except Exception:
                pass
    # Fallback: per JS entfernen
    page.evaluate("document.querySelectorAll('.cky-overlay,.cky-consent-container').forEach(e=>e.remove())")


def debug_page(page, label: str):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(DEBUG_DIR / f'{label}.png'), full_page=True)
    (DEBUG_DIR / f'{label}.html').write_text(page.content(), encoding='utf-8')
    logger.info(f'Debug: {label}.*')


def find_issue_id_in_json(data, depth=0) -> str | None:
    """Sucht rekursiv nach einer Issue-ID in JSON-Daten."""
    if depth > 8:
        return None
    if isinstance(data, dict):
        for key in ('id', 'issueId', 'issue_id', 'issueID', 'publicationId'):
            v = data.get(key)
            if isinstance(v, (int, str)) and re.match(r'^\d{5,}$', str(v)):
                return str(v)
        for v in data.values():
            r = find_issue_id_in_json(v, depth + 1)
            if r:
                return r
    elif isinstance(data, list):
        for item in data[:5]:
            r = find_issue_id_in_json(item, depth + 1)
            if r:
                return r
    return None


def run(date_str: str, output_path: Path) -> bool:
    edition_url = f'{BASE_URL}/{PUBLICATION}/{date_str}'
    issue_id    = None

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            accept_downloads=True,
        )
        page = ctx.new_page()

        # ── Netzwerk-Requests belauschen ────────────────────────────────────
        captured_requests = []

        def on_response(resp):
            url  = resp.url
            # Nur abo.merkur.de-Antworten für Issue-ID auswerten
            is_own_domain = 'abo.merkur.de' in url
            # Issue-ID direkt in URL
            for pat in [r'/download/issue/(\d{5,})', r'/issues?/(\d{5,})',
                        r'[?&]issue[Ii]d=(\d{5,})']:
                m = re.search(pat, url)
                if m:
                    captured_requests.append(('url', url, m.group(1), is_own_domain))
            # JSON-Antworten auf Inhalt prüfen (nur eigene Domain)
            if not is_own_domain:
                return
            ct = resp.headers.get('content-type', '')
            if 'json' in ct and resp.status == 200:
                try:
                    body = resp.json()
                    iid  = find_issue_id_in_json(body)
                    if iid:
                        captured_requests.append(('json', url, iid, True))
                except Exception:
                    pass

        page.on('response', on_response)

        # ── 1. Login ────────────────────────────────────────────────────────
        logger.info(f'Öffne Login: {LOGIN_URL}')
        page.goto(LOGIN_URL, wait_until='domcontentloaded')
        dismiss_cookie_banner(page)

        page.wait_for_selector('input[name="email"]', timeout=20_000)

        page.locator('input[name="email"]').fill(USER)
        page.locator('input[type="password"]').fill(PASSWD)
        dismiss_cookie_banner(page)
        page.locator('input[type="submit"]').click()

        try:
            page.wait_for_load_state('networkidle', timeout=20_000)
        except PlaywrightTimeout:
            pass

        # Login prüfen
        if page.locator('input[type="password"]').count() > 0:
            logger.error(f'Login fehlgeschlagen – URL: {page.url}')
            debug_page(page, '01_login_fail')
            browser.close()
            return False

        logger.info(f'✓ Login OK  (URL: {page.url})')

        # ── 2. Ausgaben-Seite laden + Requests abfangen ─────────────────────
        logger.info(f'Lade Ausgabe: {edition_url}')
        captured_requests.clear()
        page.goto(edition_url, wait_until='networkidle', timeout=30_000)

        # Etwas warten damit SPA vollständig geladen ist
        page.wait_for_timeout(3_000)
        debug_page(page, '02_edition_page')

        # Alle abgefangenen Requests mit Issue-ID ausgeben
        if captured_requests:
            logger.info(f'API-Requests mit Issue-ID ({len(captured_requests)}):')
            for kind, url, iid, own in captured_requests:
                domain_tag = 'own' if own else 'extern'
                logger.info(f'  [{kind}/{domain_tag}] {iid} ← {url}')

            # Beste ID: abo.merkur.de download-URL hat höchste Priorität
            for kind, url, iid, own in captured_requests:
                if own and kind == 'url' and 'download' in url:
                    issue_id = iid
                    break
            # 2. Priorität: beliebige abo.merkur.de URL
            if not issue_id:
                for kind, url, iid, own in captured_requests:
                    if own and kind == 'url':
                        issue_id = iid
                        break
            # 3. Priorität: abo.merkur.de JSON
            if not issue_id:
                for kind, url, iid, own in captured_requests:
                    if own and kind == 'json':
                        issue_id = iid
                        break
            # 4. letzter Ausweg: externe Requests (wahrscheinlich falsch)
            if not issue_id:
                logger.warning('Kein abo.merkur.de Request mit Issue-ID – nutze externen Fallback')
                issue_id = captured_requests[0][2]
        else:
            logger.warning('Keine API-Requests mit Issue-ID abgefangen')

        # Fallback 1: direkt im HTML nach Download-Links suchen
        if not issue_id:
            content = page.content()
            for pat in [r'/download/issue/(\d{5,})', r'"issueId"\s*:\s*"?(\d{5,})"?',
                        r'webreader[^"\']+#/(\d{5,})/', r'issue[/_](\d{5,})']:
                m = re.search(pat, content)
                if m:
                    issue_id = m.group(1)
                    logger.info(f'Issue-ID im HTML gefunden (Pattern: {pat}): {issue_id}')
                    break

        # Fallback 2: Download-Button klicken und URL abfangen
        if not issue_id:
            logger.info('Versuche Download-Button zu klicken...')
            for sel in ['a[href*="download/issue"]', 'a[href*="/download"]',
                        'button:has-text("Download")', 'a:has-text("PDF")',
                        'a:has-text("Download")']:
                loc = page.locator(sel)
                if loc.count() > 0:
                    href = loc.first.get_attribute('href') or ''
                    m = re.search(r'/download/issue/(\d{5,})', href)
                    if m:
                        issue_id = m.group(1)
                        logger.info(f'Issue-ID aus Download-Link: {issue_id}  ({href})')
                        break

        if not issue_id:
            logger.error('Issue-ID nicht gefunden → debug/02_edition_page.html prüfen')
            browser.close()
            return False

        logger.info(f'✓ Issue-ID: {issue_id}')

        # ── 3. PDF herunterladen ────────────────────────────────────────────
        download_url = f'{BASE_URL}/download/issue/{issue_id}'
        logger.info(f'Download: {download_url}')

        with page.expect_download(timeout=120_000) as dl_info:
            page.goto(download_url)

        dl = dl_info.value
        dl.save_as(output_path)
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f'✓ Gespeichert: {output_path} ({size_mb:.1f} MB)')

        browser.close()

    return True


def main():
    today    = datetime.now()
    date_str = today.strftime('%d.%m.%Y')
    fname    = today.strftime('%Y-%m-%d')
    output   = DOWNLOAD_DIR / f'toelzer-kurier-{fname}.pdf'

    if output.exists():
        logger.info(f'Bereits vorhanden: {output}')
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if not run(date_str, output):
        output.unlink(missing_ok=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
