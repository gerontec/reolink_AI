<?php
header('Content-Type: application/json');

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
        [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]
    );
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

$person_id = intval($input['person_id'] ?? 0);
$person_name = trim($input['person_name'] ?? '');

if ($person_id <= 0 || empty($person_name)) {
    echo json_encode(['success' => false, 'error' => 'Ungültige Parameter']);
    exit;
}

try {
    // Insert or Update
    $sql = "
        INSERT INTO person_names (object_id, object_type, person_name)
        VALUES (?, 'person', ?)
        ON DUPLICATE KEY UPDATE person_name = VALUES(person_name), updated_at = NOW()
    ";
    
    $stmt = $pdo->prepare($sql);
    $stmt->execute([$person_id, $person_name]);
    
    echo json_encode([
        'success' => true,
        'person_id' => $person_id,
        'person_name' => $person_name
    ]);
    
} catch (PDOException $e) {
    http_response_code(500);
    echo json_encode(['success' => false, 'error' => $e->getMessage()]);
}
