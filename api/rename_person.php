<?php
// Debug-Modus: Zeige RAW Output
$debug = isset($_GET['debug']) && $_GET['debug'] === '1';

if ($debug) {
    // Alle Fehler anzeigen
    ini_set('display_errors', 1);
    ini_set('display_startup_errors', 1);
    error_reporting(E_ALL);

    // Output Buffering starten
    ob_start();
}

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

// Debug-Ausgabe am Ende
if ($debug) {
    $output = ob_get_clean();

    // Zeige RAW Output mit Byte-Analyse
    header('Content-Type: text/html; charset=utf-8');
    echo "<h2>🔍 DEBUG: Raw Output Analyse</h2>";
    echo "<h3>Output Length: " . strlen($output) . " bytes</h3>";

    // Zeige erste 100 Bytes in Hex
    echo "<h3>Erste Bytes (Hex):</h3><pre>";
    for ($i = 0; $i < min(100, strlen($output)); $i++) {
        printf("%02X ", ord($output[$i]));
        if (($i + 1) % 16 === 0) echo "\n";
    }
    echo "</pre>";

    // Zeige ersten 500 Zeichen als Text
    echo "<h3>Erste 500 Zeichen:</h3><pre style='background:#f0f0f0;padding:10px;border:1px solid #ccc;'>";
    echo htmlspecialchars(substr($output, 0, 500));
    echo "</pre>";

    // Zeige kompletten Output
    echo "<h3>Kompletter Output:</h3><pre style='background:#fff;padding:10px;border:1px solid #000;'>";
    echo htmlspecialchars($output);
    echo "</pre>";

    // Prüfe auf BOM
    if (substr($output, 0, 3) === "\xEF\xBB\xBF") {
        echo "<h3 style='color:red;'>⚠️ UTF-8 BOM gefunden!</h3>";
    }

    // Prüfe auf Whitespace am Anfang
    if ($output !== '' && !in_array($output[0], ['{', '['])) {
        echo "<h3 style='color:red;'>⚠️ Output startet NICHT mit { oder [</h3>";
        echo "<p>Erstes Zeichen: '" . htmlspecialchars($output[0]) . "' (ASCII: " . ord($output[0]) . ")</p>";
    }

    exit;
}
