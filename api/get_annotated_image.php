<?php
/**
 * API: Liefert den Pfad zum annotierten Bild für ein Gesicht
 * Das annotierte Bild ist pro Recording, nicht pro Gesicht
 */

header('Content-Type: application/json');

require_once __DIR__ . '/../config.php';

try {
    $pdo = getDbConnection();
} catch (PDOException $e) {
    http_response_code(500);
    echo json_encode(['success' => false, 'error' => 'Datenbankverbindung fehlgeschlagen']);
    exit;
}

$face_id = isset($_GET['face_id']) ? intval($_GET['face_id']) : 0;

if ($face_id <= 0) {
    http_response_code(400);
    echo json_encode(['success' => false, 'error' => 'Ungültige Face ID']);
    exit;
}

try {
    // Hole das annotierte Bild vom zugehörigen Recording
    $sql = "
        SELECT r.annotated_image_path
        FROM cam2_detected_faces f
        JOIN cam2_recordings r ON f.recording_id = r.id
        WHERE f.id = ?
    ";

    $stmt = $pdo->prepare($sql);
    $stmt->execute([$face_id]);
    $result = $stmt->fetch();

    if (!$result) {
        http_response_code(404);
        echo json_encode(['success' => false, 'error' => 'Gesicht nicht gefunden']);
        exit;
    }

    if ($result['annotated_image_path']) {
        echo json_encode([
            'success' => true,
            'annotated_image_path' => $result['annotated_image_path'],
            'full_url' => '/' . $result['annotated_image_path']
        ]);
    } else {
        echo json_encode([
            'success' => false,
            'error' => 'Kein annotiertes Bild verfügbar'
        ]);
    }

} catch (PDOException $e) {
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Datenbankfehler: ' . $e->getMessage()
    ]);
}
