<?php
/**
 * API: Gesichter löschen
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

// Nur POST erlauben
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['success' => false, 'error' => 'Method not allowed']);
    exit;
}

// JSON-Daten einlesen
$input = json_decode(file_get_contents('php://input'), true);

if (!$input || !isset($input['face_ids'])) {
    http_response_code(400);
    echo json_encode(['success' => false, 'error' => 'Ungültige Parameter']);
    exit;
}

$face_ids = $input['face_ids'];

// Validierung
if (empty($face_ids) || !is_array($face_ids)) {
    echo json_encode(['success' => false, 'error' => 'Keine Face IDs angegeben']);
    exit;
}

// SQL-Injection-Schutz: Platzhalter erstellen
$placeholders = implode(',', array_fill(0, count($face_ids), '?'));

try {
    // Transaktion starten
    $pdo->beginTransaction();
    
    // Delete durchführen
    $sql = "DELETE FROM cam_detected_faces WHERE id IN ($placeholders)";
    $stmt = $pdo->prepare($sql);
    $stmt->execute($face_ids);
    
    $deleted = $stmt->rowCount();
    
    // Commit
    $pdo->commit();
    
    echo json_encode([
        'success' => true,
        'deleted' => $deleted
    ]);
    
} catch (PDOException $e) {
    $pdo->rollBack();
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Fehler beim Löschen: ' . $e->getMessage()
    ]);
}
