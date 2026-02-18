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
        captured_requests = []   # (kind, url, iid, is_own_domain)
        all_own_responses = []   # alle abo.merkur.de Response-URLs (für Debug)

        def on_response(resp):
            url  = resp.url
            is_own_domain = 'abo.merkur.de' in url
            if is_own_domain:
                all_own_responses.append((resp.status, url))
            # Issue-ID direkt in URL
            for pat in [r'/download/issue/(\d{5,})', r'/issues?/(\d{5,})',
                        r'[?&]issue[Ii]d=(\d{5,})']:
                m = re.search(pat, url)
                if m:
                    captured_requests.append(('url', url, m.group(1), is_own_domain))
            # JSON-Antworten auf Inhalt prüfen
            ct = resp.headers.get('content-type', '')
            if 'json' in ct and resp.status == 200:
                try:
                    body = resp.json()
                    iid  = find_issue_id_in_json(body)
                    if iid:
                        captured_requests.append(('json', url, iid, is_own_domain))
                    # Alle abo.merkur.de JSON-Bodies speichern
                    if is_own_domain:
                        safe = re.sub(r'[^a-zA-Z0-9]', '_', url[len('https://abo.merkur.de'):])[:60]
                        (DEBUG_DIR / f'api_{safe}.json').write_text(
                            json.dumps(body, indent=2, ensure_ascii=False), encoding='utf-8')
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
        all_own_responses.clear()
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        try:
            page.goto(edition_url, wait_until='networkidle', timeout=30_000)
        except PlaywrightTimeout:
            logger.warning('networkidle timeout – fahre fort')
        page.wait_for_timeout(3_000)
        debug_page(page, '02_edition_page')

        # Alle abo.merkur.de Responses loggen (Debug)
        logger.info(f'abo.merkur.de Responses ({len(all_own_responses)}):')
        for status, u in all_own_responses:
            logger.info(f'  {status} {u}')

        def pick_best_id():
            # Prio 1: abo.merkur.de URL mit /download/
            for kind, u, iid, own in captured_requests:
                if own and kind == 'url' and 'download' in u:
                    return iid
            # Prio 2: beliebige abo.merkur.de URL
            for kind, u, iid, own in captured_requests:
                if own and kind == 'url':
                    return iid
            # Prio 3: abo.merkur.de JSON
            for kind, u, iid, own in captured_requests:
                if own and kind == 'json':
                    return iid
            return None

        if captured_requests:
            logger.info(f'API-Requests mit Issue-ID ({len(captured_requests)}):')
            for kind, u, iid, own in captured_requests:
                logger.info(f'  [{kind}/{"own" if own else "extern"}] {iid} ← {u}')
            issue_id = pick_best_id()
            if not issue_id:
                logger.warning('Kein abo.merkur.de Request mit Issue-ID')
        else:
            logger.warning('Keine API-Requests mit Issue-ID abgefangen')

        # Fallback 1: alle <a href> Links auf der Seite ausgeben + durchsuchen
        if not issue_id:
            hrefs = page.eval_on_selector_all('a[href]', 'els => els.map(e => e.getAttribute("href"))')
            dl_links = [h for h in hrefs if h and '/download/issue/' in h]
            logger.info(f'Download-Links auf Seite: {dl_links}')
            if dl_links:
                m = re.search(r'/download/issue/(\d{5,})', dl_links[0])
                if m:
                    issue_id = m.group(1)
                    logger.info(f'Issue-ID aus Download-Link: {issue_id}')
            # Alle Links loggen (erste 30)
            logger.info(f'Alle Links ({len(hrefs)} gesamt, erste 30): {hrefs[:30]}')

        # Fallback 2: Cover-Image klicken → Webreader-URL enthält Issue-ID
        # Flow: Klick auf Cover-Bild (s4p-iapps.com) → Navigation zu
        #       abo.merkur.de/webreader-v3/index.html#/895556/1-
        # Hash-Fragment-Navigation erzeugt keinen HTTP-Request → Interceptor greift nicht!
        if not issue_id:
            cover_sel = 'img[src*="s4p-iapps.com"], img[src*="s4p"]'
            cover_loc = page.locator(cover_sel)
            if cover_loc.count() > 0:
                logger.info(f'Cover-Image gefunden ({cover_loc.count()}x) – prüfe Parent-href...')
                # Zunächst Parent-<a> href prüfen
                parent = cover_loc.first.locator('xpath=..')
                href = parent.get_attribute('href') or ''
                logger.info(f'Parent href: {href!r}')
                m = re.search(r'[/#](\d{5,})', href)
                if m:
                    issue_id = m.group(1)
                    logger.info(f'Issue-ID aus Parent-href: {issue_id}')
                else:
                    # Klick – kann same-page Navigation oder neues Tab öffnen
                    logger.info('Klicke Cover-Image...')
                    try:
                        with ctx.expect_page(timeout=5_000) as new_page_info:
                            cover_loc.first.click()
                        new_url = new_page_info.value.url
                        logger.info(f'Neues Tab: {new_url}')
                    except PlaywrightTimeout:
                        # Kein neues Tab → same-page Navigation, kurz auf URL-Änderung warten
                        page.wait_for_timeout(2_000)
                        new_url = page.url
                        logger.info(f'Same-page nach Klick: {new_url}')
                    m = re.search(r'[/#](\d{5,})', new_url)
                    if m:
                        issue_id = m.group(1)
                        logger.info(f'Issue-ID aus Navigation-URL: {issue_id}')
            else:
                logger.info('Kein Cover-Image (s4p-iapps.com) auf Seite gefunden')

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
