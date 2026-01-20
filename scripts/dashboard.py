#!/home/gh/python/venv_py311/bin/python3
import pymysql
import os
from datetime import datetime

# ─── KONFIGURATION ────────────────────────────────────────────────
DB_CONFIG = {
    'host': '192.168.178.218',
    'user': 'gh',
    'password': 'a12345',
    'database': 'wagodb'
}
WEB_VIDEO_PATH = "/aufnahmen" 
REPORT_DIR = "/var/www/html/reports"

def get_data(sql):
    try:
        conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    except Exception as e:
        print(f"❌ SQL Fehler: {e}")
        return []
    finally:
        if 'conn' in locals(): conn.close()

def generate_report():
    print(f"📊 [{datetime.now().strftime('%H:%M:%S')}] Erstelle Dashboard...")

    # 1. Haupt-Statistiken
    v_sum = get_data("SELECT COUNT(*) as total, ROUND(SUM(filesize_mb)/1024, 2) as gb, SUM(total_detections) as det FROM cam_video_archive")[0]
    f_sum = get_data("SELECT COUNT(*) as total, SUM(CASE WHEN recognized=1 THEN 1 ELSE 0 END) as rec FROM cam_face_recognitions")[0]
    
    # 2. Letzte Videos
    recent_videos = get_data("SELECT recorded_at, filename, trigger_object_type, filesize_mb FROM cam_video_archive ORDER BY recorded_at DESC LIMIT 15")
    
    # 3. Objekt-Analyse
    obj_analysis = get_data("SELECT object_type, COUNT(*) as count, SUM(times_crossed_line) as crossings FROM cam_detected_objects GROUP BY object_type ORDER BY count DESC")
    
    # 4. Geparkte Autos - Langzeit (>2h, >1000 detections)
    parked_cars = get_data("""
        SELECT 
            object_hash,
            object_type,
            total_detections,
            times_crossed_line,
            TIMESTAMPDIFF(HOUR, first_seen, last_seen) as hours_parked,
            first_seen,
            last_seen
        FROM cam_detected_objects
        WHERE object_type IN ('car', 'truck', 'bus')
          AND total_detections > 1000
          AND times_crossed_line = 0
          AND TIMESTAMPDIFF(HOUR, first_seen, last_seen) > 2
          AND (total_detections / GREATEST(TIMESTAMPDIFF(HOUR, first_seen, last_seen), 1)) > 500
        ORDER BY total_detections DESC
        LIMIT 10
    """)
    
    # 5. Aktuelle Parkplatz-Belegung (spatial clustering)
    parked_raw = get_data("""
        SELECT 
            object_hash,
            object_type,
            total_detections,
            TIMESTAMPDIFF(MINUTE, first_seen, last_seen) as duration_minutes
        FROM cam_detected_objects
        WHERE object_type IN ('car', 'truck', 'bus')
          AND last_seen >= DATE_SUB(NOW(), INTERVAL 15 MINUTE)
          AND times_crossed_line = 0
          AND total_detections > 1000
          AND TIMESTAMPDIFF(MINUTE, first_seen, last_seen) > 60
    """)
    
    # Spatial clustering: Group similar hashes (same vehicle, different detections)
    parked_clusters = {}
    for p in parked_raw:
        # Use first 4 chars of hash as cluster key (spatial region)
        cluster_key = p['object_hash'][:2]
        if cluster_key not in parked_clusters:
            parked_clusters[cluster_key] = {
                'detections': 0,
                'duration': 0,
                'types': set()
            }
        parked_clusters[cluster_key]['detections'] += p['total_detections']
        parked_clusters[cluster_key]['duration'] = max(parked_clusters[cluster_key]['duration'], p['duration_minutes'])
        parked_clusters[cluster_key]['types'].add(p['object_type'])
    
    parked_now = len(parked_clusters)

    # 6. KI Klassifizierungen
    classifications = get_data("SELECT top1_class_name, COUNT(*) as count FROM cam_image_classification GROUP BY top1_class_name ORDER BY count DESC LIMIT 5")

    # ─── VIDEO TABELLE ────────────────────────────────────────────
    video_rows = ""
    for v in recent_videos:
        sub_folder = v['recorded_at'].strftime('%Y/%m')
        video_url = f"{WEB_VIDEO_PATH}/{sub_folder}/{v['filename']}"
        video_rows += f'''
        <tr>
            <td>{v['recorded_at'].strftime('%d.%m. %H:%M:%S')}</td>
            <td><span class="badge-trigger">{v['trigger_object_type']}</span></td>
            <td>{v['filesize_mb']} MB</td>
            <td><a href="{video_url}" target="_blank" class="play-link">▶ Video öffnen</a></td>
        </tr>'''
    
    # ─── PARKPLATZ TABELLE ────────────────────────────────────────
    parked_rows = ""
    for p in parked_cars:
        duration = f"{p['hours_parked']}h" if p['hours_parked'] < 24 else f"{p['hours_parked']//24}d {p['hours_parked']%24}h"
        parked_rows += f'''
        <tr>
            <td><code style="font-size:0.85em;">{p['object_hash'][:8]}...</code></td>
            <td><span class="type-badge">{p['object_type']}</span></td>
            <td>{p['total_detections']:,}</td>
            <td>{duration}</td>
            <td>{p['last_seen'].strftime('%d.%m. %H:%M')}</td>
        </tr>'''

    # ─── HTML DASHBOARD ───────────────────────────────────────────
    html = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="60">
        <title>Watchdog NVMe Dashboard</title>
        <style>
            :root {{ --bg: #0d1117; --card: #161b22; --text: #c9d1d9; --blue: #58a6ff; --green: #238636; --orange: #d29922; --border: #30363d; }}
            body {{ font-family: 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); padding: 20px; margin: 0; }}
            .container {{ max-width: 1400px; margin: auto; }}
            h1, h2 {{ color: var(--blue); border-bottom: 1px solid var(--border); padding-bottom: 10px; }}
            
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmin(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
            .card {{ background: var(--card); padding: 20px; border-radius: 8px; border: 1px solid var(--border); text-align: center; }}
            .card .label {{ font-size: 0.8em; color: #8b949e; text-transform: uppercase; }}
            .card .value {{ font-size: 2em; font-weight: bold; margin-top: 5px; color: #fff; }}
            .card.parking {{ border-color: var(--orange); }}
            .card.parking .value {{ color: var(--orange); }}

            .main-grid {{ display: grid; grid-template-columns: 1.5fr 1fr; gap: 20px; margin-bottom: 20px; }}
            .full-width {{ grid-column: 1 / -1; }}
            
            table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid var(--border); }}
            th {{ background: #21262d; color: var(--blue); font-size: 0.9em; }}
            
            .badge-trigger {{ background: #388bfd33; color: var(--blue); padding: 3px 8px; border-radius: 12px; font-size: 0.85em; border: 1px solid #388bfd66; }}
            .type-badge {{ background: #8b949e33; color: #8b949e; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }}
            .play-link {{ color: var(--blue); text-decoration: none; font-weight: bold; }}
            .play-link:hover {{ text-decoration: underline; }}
            
            .ai-tag {{ background: #23863633; color: #3fb950; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; margin-right: 5px; }}
            
            .parking-indicator {{
                display: inline-block;
                width: 12px;
                height: 12px;
                background: var(--orange);
                border-radius: 50%;
                margin-right: 8px;
                animation: pulse 2s infinite;
            }}
            
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Watchdog System Pro <span style="font-size: 0.4em; color: #8b949e;">NVMe Edition</span></h1>
            <p>Letztes Update: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')} | <span style="color: var(--green);">Auto-Refresh: 60s</span></p>

            <div class="stats-grid">
                <div class="card"><div class="label">Videos</div><div class="value">{v_sum['total']}</div></div>
                <div class="card"><div class="label">Speicher</div><div class="value">{v_sum['gb']} GB</div></div>
                <div class="card"><div class="label">Objekte</div><div class="value">{v_sum['det']}</div></div>
                <div class="card"><div class="label">Gesichter</div><div class="value">{f_sum['rec'] or 0}</div></div>
                <div class="card parking">
                    <div class="label">🅿️ Aktuell Geparkt</div>
                    <div class="value"><span class="parking-indicator"></span>{parked_now}</div>
                </div>
            </div>

            <div class="main-grid">
                <div class="video-section">
                    <h2>🎬 Letzte Aufnahmen (NVMe)</h2>
                    <table>
                        <thead><tr><th>Zeit</th><th>Event</th><th>Größe</th><th>Aktion</th></tr></thead>
                        <tbody>{video_rows}</tbody>
                    </table>
                </div>

                <div class="side-section">
                    <h2>📦 Objekte</h2>
                    <table>
                        <thead><tr><th>Typ</th><th>Cross</th></tr></thead>
                        <tbody>
                            {''.join([f"<tr><td>{o['object_type']}</td><td>{o['crossings']}</td></tr>" for o in obj_analysis])}
                        </tbody>
                    </table>

                    <h2>🤖 Top KI-Szenen</h2>
                    <div style="background: var(--card); padding: 15px; border-radius: 8px; border: 1px solid var(--border);">
                        {''.join([f"<span class='ai-tag'>{c['top1_class_name']} ({c['count']})</span>" for c in classifications])}
                    </div>
                </div>
            </div>
            
            <div class="full-width">
                <h2>🅿️ Geparkte Fahrzeuge (Lange Standzeit)</h2>
                <table>
                    <thead><tr><th>Fahrzeug ID</th><th>Typ</th><th>Detections</th><th>Parkdauer</th><th>Zuletzt gesehen</th></tr></thead>
                    <tbody>
                        {parked_rows if parked_rows else '<tr><td colspan="5" style="text-align:center; color: #8b949e;">Keine langzeit-geparkten Fahrzeuge</td></tr>'}
                    </tbody>
                </table>
                <p style="color: #8b949e; font-size: 0.85em; margin-top: 10px;">
                    Kriterien: >1000 Detections, 0 Line Crossings, >2h Aufenthalt | Spatial Clustering aktiv
                </p>
            </div>

            <footer style="margin-top: 50px; text-align: center; color: #484f58; font-size: 0.8em;">
                Pfad: {WEB_VIDEO_PATH}/YYYY/MM/ | DB: wagodb | Server: 192.168.178.218 | Spatial Clustering v1.0
            </footer>
        </div>
    </body>
    </html>
    """

    # ─── SPEICHERN ────────────────────────────────────────────────
    os.makedirs(REPORT_DIR, exist_ok=True)
    file_path = os.path.join(REPORT_DIR, "index.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    # Set permissions without sudo
    try:
        os.chmod(file_path, 0o664)
    except:
        pass
    
    print(f"✅ Dashboard aktualisiert: {file_path}")
    print(f"🅿️  Aktuell geparkt: {parked_now} Fahrzeug(e) (spatial clustering)")
    print(f"📋 Langzeit-Parker: {len(parked_cars)}")

if __name__ == "__main__":
    generate_report()
