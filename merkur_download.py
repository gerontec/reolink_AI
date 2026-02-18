#!/usr/bin/env python3
"""
Tölzer Kurier / Merkur - Täglicher PDF-Download
Playwright: Login + Cover-Image-href → Issue-ID → Download

Verwendung:
  python3 merkur_download.py                   # heute
  python3 merkur_download.py --date 20260218   # bestimmtes Datum

Installation:
  pip install playwright
  playwright install chromium

Cron (täglich 06:30):
  30 6 * * * /home/gh/python/venv_py311/bin/python3 /home/gh/python/merkur_download.py >> /home/gh/python/zeitung/cron.log 2>&1
"""

import re
import sys
import json
import argparse
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

        # ── Netzwerk-Requests belauschen (Fallback falls kein Cover-Image) ──
        captured_requests = []   # (kind, url, iid, is_own_domain)

        def on_response(resp):
            url  = resp.url
            is_own_domain = 'abo.merkur.de' in url
            for pat in [r'/download/issue/(\d{5,})', r'/issues?/(\d{5,})',
                        r'[?&]issue[Ii]d=(\d{5,})']:
                m = re.search(pat, url)
                if m:
                    captured_requests.append(('url', url, m.group(1), is_own_domain))
            ct = resp.headers.get('content-type', '')
            if 'json' in ct and resp.status == 200 and is_own_domain:
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

        try:
            page.goto(edition_url, wait_until='networkidle', timeout=30_000)
        except PlaywrightTimeout:
            logger.warning('networkidle timeout – fahre fort')
        page.wait_for_timeout(3_000)

        # Prio 1: abo.merkur.de API-Request mit Issue-ID (selten, aber ideal)
        for kind, u, iid, own in captured_requests:
            if own:
                issue_id = iid
                logger.info(f'Issue-ID aus API: {issue_id}  ({u})')
                break

        # Prio 2: Cover-Image Parent-href → /webreader/895556 enthält Issue-ID
        # Der Klick auf das Titelbild (s4p-iapps.com) löst Hash-Fragment-Navigation aus
        # → kein HTTP-Request → Response-Interceptor greift nicht, aber href ist im DOM.
        if not issue_id:
            cover_loc = page.locator('img[src*="s4p-iapps.com"], img[src*="s4p"]')
            if cover_loc.count() > 0:
                href = cover_loc.first.locator('xpath=..').get_attribute('href') or ''
                m = re.search(r'[/#](\d{5,})', href)
                if m:
                    issue_id = m.group(1)
                    logger.info(f'Issue-ID aus Cover-href: {issue_id}')
                else:
                    # Fallback: Cover klicken und URL abfangen
                    try:
                        with ctx.expect_page(timeout=5_000) as np_info:
                            cover_loc.first.click()
                        new_url = np_info.value.url
                    except PlaywrightTimeout:
                        page.wait_for_timeout(2_000)
                        new_url = page.url
                    m = re.search(r'[/#](\d{5,})', new_url)
                    if m:
                        issue_id = m.group(1)
                        logger.info(f'Issue-ID aus Click-Navigation: {issue_id}')
            else:
                logger.warning('Kein Cover-Image (s4p-iapps.com) gefunden')

        if not issue_id:
            logger.error('Issue-ID nicht gefunden → debug/02_edition_page.html prüfen')
            browser.close()
            return False

        logger.info(f'✓ Issue-ID: {issue_id}')

        # ── 3. PDF herunterladen ────────────────────────────────────────────
        download_url = f'{BASE_URL}/download/issue/{issue_id}'
        logger.info(f'Download: {download_url}')

        with page.expect_download(timeout=120_000) as dl_info:
            try:
                page.goto(download_url)
            except Exception:
                pass  # Playwright wirft "Download is starting" – Download läuft trotzdem

        dl = dl_info.value
        dl.save_as(output_path)
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f'✓ Gespeichert: {output_path} ({size_mb:.1f} MB)')

        browser.close()

    return True


def main():
    parser = argparse.ArgumentParser(description='Tölzer Kurier PDF-Download')
    parser.add_argument('--date', metavar='YYYYMMDD',
                        help='Datum der Ausgabe (Standard: heute)')
    args = parser.parse_args()

    if args.date:
        try:
            day = datetime.strptime(args.date, '%Y%m%d')
        except ValueError:
            logger.error('--date muss im Format YYYYMMDD angegeben werden, z.B. 20260218')
            sys.exit(1)
    else:
        day = datetime.now()

    date_str = day.strftime('%d.%m.%Y')       # für URL: 18.02.2026
    fname    = day.strftime('%Y-%m-%d')        # für Dateiname: 2026-02-18
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
