-- Parkplatz-Definitionstabelle für Stellplätze + Garagen
-- Layout: 5 Stellplätze rechts (vertikal) + 2 Garagen unten (horizontal)
-- Bildauflösung: 4512 × 2512 (4K Reolink)

CREATE TABLE IF NOT EXISTS cam2_parking_spots (
    id INT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    type ENUM('stellplatz', 'garage') NOT NULL,
    x1 INT NOT NULL,
    y1 INT NOT NULL,
    x2 INT NOT NULL,
    y2 INT NOT NULL,
    INDEX idx_type (type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Alte Einträge löschen falls vorhanden
TRUNCATE TABLE cam2_parking_spots;

-- Stellplätze 1-5 (rechte 20% des Screens, vertikal aufgeteilt)
-- x: 80%-100% (3610-4512)
-- y: je 20% (0-502, 502-1004, 1004-1507, 1507-2009, 2009-2512)
INSERT INTO cam2_parking_spots (id, name, type, x1, y1, x2, y2) VALUES
(1, 'Stellplatz 1', 'stellplatz', 3610, 0,    4512, 502),
(2, 'Stellplatz 2', 'stellplatz', 3610, 502,  4512, 1004),
(3, 'Stellplatz 3', 'stellplatz', 3610, 1004, 4512, 1507),
(4, 'Stellplatz 4', 'stellplatz', 3610, 1507, 4512, 2009),
(5, 'Stellplatz 5', 'stellplatz', 3610, 2009, 4512, 2512),

-- Garagen 6-7 (untere 10% des Screens, horizontal aufgeteilt)
-- x: je 50% (0-2256, 2256-4512)
-- y: 90%-100% (2261-2512)
(6, 'Garage Links',  'garage', 0,    2261, 2256, 2512),
(7, 'Garage Rechts', 'garage', 2256, 2261, 4512, 2512);

-- Ansicht der Parkplätze
SELECT
    id,
    name,
    type,
    CONCAT(x1, '-', x2, ' × ', y1, '-', y2) as koordinaten,
    (x2-x1) * (y2-y1) as flaeche_pixel
FROM cam2_parking_spots
ORDER BY id;
