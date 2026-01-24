#!/home/gh/python/venv_py311/bin/python3
"""
Erstellt cam2_parking_spots Tabelle mit korrekten Parkplatz-Koordinaten
Layout: 5 Stellplätze (rechts vertikal) + 2 Garagen (unten horizontal)
"""

import pymysql

DB_CONFIG = {
    'host': 'localhost',
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

def setup_parking_spots():
    """Erstellt Parkplatz-Tabelle und fügt Definitionen ein"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        print("Erstelle cam2_parking_spots Tabelle...")

        # Tabelle erstellen
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cam2_parking_spots (
                id INT PRIMARY KEY,
                name VARCHAR(50) NOT NULL,
                type ENUM('stellplatz', 'garage') NOT NULL,
                x1 INT NOT NULL,
                y1 INT NOT NULL,
                x2 INT NOT NULL,
                y2 INT NOT NULL,
                INDEX idx_type (type)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # Alte Einträge löschen
        cursor.execute("DELETE FROM cam2_parking_spots")

        # Parkplatz-Definitionen
        spots = [
            # Stellplätze 1-5 (rechts, vertikal)
            (1, 'Stellplatz 1', 'stellplatz', 3610, 0,    4512, 502),
            (2, 'Stellplatz 2', 'stellplatz', 3610, 502,  4512, 1004),
            (3, 'Stellplatz 3', 'stellplatz', 3610, 1004, 4512, 1507),
            (4, 'Stellplatz 4', 'stellplatz', 3610, 1507, 4512, 2009),
            (5, 'Stellplatz 5', 'stellplatz', 3610, 2009, 4512, 2512),
            # Garagen 6-7 (unten, horizontal)
            (6, 'Garage Links',  'garage', 0,    2261, 2256, 2512),
            (7, 'Garage Rechts', 'garage', 2256, 2261, 4512, 2512),
        ]

        # Einfügen
        cursor.executemany("""
            INSERT INTO cam2_parking_spots (id, name, type, x1, y1, x2, y2)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, spots)

        conn.commit()

        # Anzeigen
        cursor.execute("""
            SELECT
                id,
                name,
                type,
                CONCAT(x1, '-', x2, ' × ', y1, '-', y2) as koordinaten,
                (x2-x1) * (y2-y1) as flaeche_pixel
            FROM cam2_parking_spots
            ORDER BY id
        """)

        print("\n✓ Parkplätze erfolgreich angelegt:\n")
        print(f"{'ID':<5} {'Name':<20} {'Typ':<12} {'Koordinaten':<25} {'Fläche (px²)':<15}")
        print("-" * 80)

        for row in cursor.fetchall():
            print(f"{row[0]:<5} {row[1]:<20} {row[2]:<12} {row[3]:<25} {row[4]:>15,}")

        cursor.close()
        conn.close()

        print("\n✓ Setup abgeschlossen!")
        print("\nHinweis: Führe watchdog2.py mit --reanalyze aus um bestehende Aufnahmen neu zu analysieren.")

    except Exception as e:
        print(f"✗ Fehler: {e}")
        return False

    return True


if __name__ == "__main__":
    setup_parking_spots()
