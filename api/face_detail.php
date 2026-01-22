<?php
/**
 * API: Detaillierte Informationen zu einem Gesicht
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

// Parameter
$face_id = isset($_GET['id']) ? intval($_GET['id']) : 0;

if ($face_id <= 0) {
    http_response_code(400);
    echo json_encode(['success' => false, 'error' => 'Ungültige Face ID']);
    exit;
}

try {
    // Detaillierte Informationen abrufen
    $sql = "
        SELECT 
            f.id,
            f.person_name,
            f.confidence,
            f.bbox_x1,
            f.bbox_y1,
            f.bbox_x2,
            f.bbox_y2,
            f.detected_at,
            (f.bbox_x2 - f.bbox_x1) as width,
            (f.bbox_y2 - f.bbox_y1) as height,
            r.file_path,
            r.file_size,
            r.camera_name,
            r.recorded_at,
            r.file_type
        FROM 
            cam_detected_faces f
        JOIN 
            cam_recordings r ON f.recording_id = r.id
        WHERE 
            f.id = ?
    ";
    
    $stmt = $pdo->prepare($sql);
    $stmt->execute([$face_id]);
    $face = $stmt->fetch();
    
    if (!$face) {
        http_response_code(404);
        echo json_encode(['success' => false, 'error' => 'Gesicht nicht gefunden']);
        exit;
    }
    
    // Formatierung
    $face['recorded_at'] = date('d.m.Y H:i:s', strtotime($face['recorded_at']));
    $face['detected_at'] = date('d.m.Y H:i:s', strtotime($face['detected_at']));
    $face['file_size_mb'] = round($face['file_size'] / 1024 / 1024, 2);
    $face['face_area'] = $face['width'] * $face['height'];
    
    echo json_encode([
        'success' => true,
        'face' => $face
    ]);
    
} catch (PDOException $e) {
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Datenbankfehler: ' . $e->getMessage()
    ]);
}
