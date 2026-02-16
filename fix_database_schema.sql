-- Fix Database Schema für parking_spot_id und scene_category
-- Ausführen mit: mysql -u gh -pa12345 wagodb < fix_database_schema.sql

USE wagodb;

-- 1. Prüfe und füge parking_spot_id zu cam2_detected_objects hinzu
ALTER TABLE cam2_detected_objects
ADD COLUMN parking_spot_id INT DEFAULT NULL AFTER bbox_y2;

-- 2. Index für parking_spot_id
ALTER TABLE cam2_detected_objects
ADD INDEX idx_parking_spot (parking_spot_id);

-- 3. Prüfe und füge scene_category zu cam2_analysis_summary hinzu
ALTER TABLE cam2_analysis_summary
ADD COLUMN scene_category VARCHAR(50) DEFAULT 'unknown' AFTER max_persons;

-- 4. Index für scene_category
ALTER TABLE cam2_analysis_summary
ADD INDEX idx_scene (scene_category);

-- 5. Erstelle cam2_parking_stats Tabelle
CREATE TABLE IF NOT EXISTS cam2_parking_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    parking_spot_id INT NOT NULL,
    date DATE NOT NULL,
    hour INT NOT NULL,
    occupancy_count INT DEFAULT 0,
    vehicle_type VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_spot_time (parking_spot_id, date, hour, vehicle_type),
    INDEX idx_spot (parking_spot_id),
    INDEX idx_date (date),
    INDEX idx_spot_date (parking_spot_id, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Zeige Ergebnis
SHOW COLUMNS FROM cam2_detected_objects LIKE '%parking%';
SHOW COLUMNS FROM cam2_analysis_summary LIKE '%scene%';
SHOW TABLES LIKE 'cam2_parking_stats';
