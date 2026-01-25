<?php
/**
 * API: Gesichter umbenennen
 */

header('Content-Type: application/json');

// Database Configuration
$db_config = [
    'host' => 'localhost',
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

if (!$input || !isset($input['face_ids']) || !isset($input['new_name'])) {
    http_response_code(400);
    echo json_encode(['success' => false, 'error' => 'Ungültige Parameter']);
    exit;
}

$face_ids = $input['face_ids'];
$new_name = trim($input['new_name']);

// Validierung
if (empty($face_ids) || !is_array($face_ids)) {
    echo json_encode(['success' => false, 'error' => 'Keine Face IDs angegeben']);
    exit;
}

if (empty($new_name)) {
    echo json_encode(['success' => false, 'error' => 'Kein Name angegeben']);
    exit;
}

if (strlen($new_name) > 100) {
    echo json_encode(['success' => false, 'error' => 'Name zu lang (max. 100 Zeichen)']);
    exit;
}

// SQL-Injection-Schutz: Platzhalter erstellen
$placeholders = implode(',', array_fill(0, count($face_ids), '?'));

try {
    // Transaktion starten
    $pdo->beginTransaction();
    
    // Update durchführen
    $sql = "UPDATE cam2_detected_faces SET person_name = ? WHERE id IN ($placeholders)";
    $stmt = $pdo->prepare($sql);
    
    // Parameter zusammenführen: neuer Name + alle Face IDs
    $params = array_merge([$new_name], $face_ids);
    $stmt->execute($params);
    
    $updated = $stmt->rowCount();
    
    // Commit
    $pdo->commit();
    
    echo json_encode([
        'success' => true,
        'updated' => $updated,
        'new_name' => $new_name
    ]);
    
} catch (PDOException $e) {
    $pdo->rollBack();
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Fehler beim Update: ' . $e->getMessage()
    ]);
}
