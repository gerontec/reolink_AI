<?php
/**
 * API: Statistiken und Dashboard-Daten
 */

header('Content-Type: application/json');

// Database Configuration
$db_config = [
    'host' => '192.168.178.218',
    'database' => 'wagodb',
    'user' => 'gh',
    'password' => 'a12345',
    'charset' => 'utf8mb4'
];

try {
    $pdo = new PDO(
        "mysql:host={$db_config['host']};dbname={$db_config['database']};charset={$db_config['charset']}",
        $db_config['user'],
        $db_config['password'],
        [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC
        ]
    );
} catch (PDOException $e) {
    http_response_code(500);
    echo json_encode(['success' => false, 'error' => 'Datenbankverbindung fehlgeschlagen']);
    exit;
}

try {
    // Gesamt-Statistiken
    $stats = [];
    
    // Gesichter
    $stats['total_faces'] = $pdo->query("SELECT COUNT(*) FROM cam_detected_faces")->fetchColumn();
    $stats['unknown_faces'] = $pdo->query("SELECT COUNT(*) FROM cam_detected_faces WHERE person_name = 'Unknown'")->fetchColumn();
    $stats['named_persons'] = $pdo->query("SELECT COUNT(DISTINCT person_name) FROM cam_detected_faces WHERE person_name != 'Unknown'")->fetchColumn();
    
    // Aufnahmen
    $stats['total_recordings'] = $pdo->query("SELECT COUNT(*) FROM cam_recordings")->fetchColumn();
    $stats['total_images'] = $pdo->query("SELECT COUNT(*) FROM cam_recordings WHERE file_type = 'jpg'")->fetchColumn();
    $stats['total_videos'] = $pdo->query("SELECT COUNT(*) FROM cam_recordings WHERE file_type = 'mp4'")->fetchColumn();
    
    // Objekte
    $stats['total_objects'] = $pdo->query("SELECT COUNT(*) FROM cam_detected_objects")->fetchColumn();
    $stats['total_vehicles'] = $pdo->query("SELECT COUNT(*) FROM cam_detected_objects WHERE object_class IN ('car', 'truck', 'bus', 'motorcycle', 'bicycle')")->fetchColumn();
    
    // Top Personen
    $top_persons = $pdo->query("
        SELECT person_name, COUNT(*) as count 
        FROM cam_detected_faces 
        WHERE person_name != 'Unknown'
        GROUP BY person_name 
        ORDER BY count DESC 
        LIMIT 10
    ")->fetchAll();
    $stats['top_persons'] = $top_persons;
    
    // Top Objekte
    $top_objects = $pdo->query("
        SELECT object_class, COUNT(*) as count 
        FROM cam_detected_objects 
        GROUP BY object_class 
        ORDER BY count DESC 
        LIMIT 10
    ")->fetchAll();
    $stats['top_objects'] = $top_objects;
    
    // Aktivität pro Kamera
    $camera_activity = $pdo->query("
        SELECT camera_name, COUNT(*) as recordings 
        FROM cam_recordings 
        GROUP BY camera_name 
        ORDER BY recordings DESC
    ")->fetchAll();
    $stats['camera_activity'] = $camera_activity;
    
    // Zeitliche Verteilung (letzte 7 Tage)
    $daily_stats = $pdo->query("
        SELECT 
            DATE(recorded_at) as date,
            COUNT(*) as recordings,
            SUM(CASE WHEN file_type = 'jpg' THEN 1 ELSE 0 END) as images,
            SUM(CASE WHEN file_type = 'mp4' THEN 1 ELSE 0 END) as videos
        FROM cam_recordings
        WHERE recorded_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY DATE(recorded_at)
        ORDER BY date DESC
    ")->fetchAll();
    $stats['daily_stats'] = $daily_stats;
    
    echo json_encode([
        'success' => true,
        'stats' => $stats
    ]);
    
} catch (PDOException $e) {
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Fehler beim Abrufen der Statistiken: ' . $e->getMessage()
    ]);
}
