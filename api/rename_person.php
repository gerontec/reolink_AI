<?php
header('Content-Type: application/json');

require_once __DIR__ . '/../config.php';

try {
    $pdo = getDbConnection();
} catch (PDOException $e) {
    http_response_code(500);
    echo json_encode(['success' => false, 'error' => 'DB-Fehler']);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['success' => false, 'error' => 'Method not allowed']);
    exit;
}

$input = json_decode(file_get_contents('php://input'), true);

$face_id = intval($input['face_id'] ?? 0);
$person_name = trim($input['person_name'] ?? '');

if ($face_id <= 0 || empty($person_name)) {
    echo json_encode(['success' => false, 'error' => 'Ungültige Parameter']);
    exit;
}

try {
    // Update person_name in cam2_detected_faces
    $sql = "
        UPDATE cam2_detected_faces
        SET person_name = ?
        WHERE id = ?
    ";

    $stmt = $pdo->prepare($sql);
    $stmt->execute([$person_name, $face_id]);

    if ($stmt->rowCount() === 0) {
        echo json_encode(['success' => false, 'error' => 'Gesicht nicht gefunden']);
        exit;
    }

    echo json_encode([
        'success' => true,
        'face_id' => $face_id,
        'person_name' => $person_name
    ]);

} catch (PDOException $e) {
    http_response_code(500);
    echo json_encode(['success' => false, 'error' => $e->getMessage()]);
}
