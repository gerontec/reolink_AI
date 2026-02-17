<?php
/**
 * API: Gesicht aus Bild ausschneiden und zurückgeben
 */

require_once __DIR__ . '/../config.php';

try {
    $pdo = getDbConnection();
} catch (PDOException $e) {
    http_response_code(500);
    echo json_encode(['success' => false, 'error' => 'Datenbankverbindung fehlgeschlagen']);
    exit;
}

// Parameter
$face_id = isset($_GET['id']) ? intval($_GET['id']) : 0;
$size = isset($_GET['size']) ? intval($_GET['size']) : 200;
$padding = isset($_GET['padding']) ? intval($_GET['padding']) : 20;

if ($face_id <= 0) {
    http_response_code(400);
    header('Content-Type: text/plain');
    echo 'Ungültige Face ID';
    exit;
}

// Größe limitieren
$size = min(max($size, 50), 800);

try {
    // Gesicht-Daten abrufen
    $sql = "
        SELECT
            f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
            r.file_path
        FROM
            cam2_detected_faces f
        JOIN
            cam2_recordings r ON f.recording_id = r.id
        WHERE 
            f.id = ?
    ";
    
    $stmt = $pdo->prepare($sql);
    $stmt->execute([$face_id]);
    $face = $stmt->fetch();
    
    if (!$face) {
        http_response_code(404);
        header('Content-Type: text/plain');
        echo 'Gesicht nicht gefunden';
        exit;
    }
    
    // Bildpfad
    $base_path = '/var/www/web1/';
    $image_path = $base_path . $face['file_path'];
    
    if (!file_exists($image_path)) {
        http_response_code(404);
        header('Content-Type: text/plain');
        echo 'Bild nicht gefunden: ' . $image_path;
        exit;
    }
    
    // Bild laden
    $image = imagecreatefromjpeg($image_path);
    if (!$image) {
        http_response_code(500);
        header('Content-Type: text/plain');
        echo 'Bild konnte nicht geladen werden';
        exit;
    }
    
    $img_width = imagesx($image);
    $img_height = imagesy($image);
    
    // Bounding Box mit Padding
    $x1 = max(0, $face['bbox_x1'] - $padding);
    $y1 = max(0, $face['bbox_y1'] - $padding);
    $x2 = min($img_width, $face['bbox_x2'] + $padding);
    $y2 = min($img_height, $face['bbox_y2'] + $padding);
    
    $crop_width = $x2 - $x1;
    $crop_height = $y2 - $y1;
    
    if ($crop_width <= 0 || $crop_height <= 0) {
        http_response_code(500);
        header('Content-Type: text/plain');
        echo 'Ungültige Crop-Dimensionen';
        exit;
    }
    
    // Ausschnitt erstellen
    $cropped = imagecrop($image, [
        'x' => $x1,
        'y' => $y1,
        'width' => $crop_width,
        'height' => $crop_height
    ]);
    
    if (!$cropped) {
        http_response_code(500);
        header('Content-Type: text/plain');
        echo 'Crop fehlgeschlagen';
        exit;
    }
    
    // Auf gewünschte Größe skalieren
    $resized = imagescale($cropped, $size, $size);
    
    // Cache-Header
    header('Content-Type: image/jpeg');
    header('Cache-Control: public, max-age=86400'); // 24 Stunden
    header('Expires: ' . gmdate('D, d M Y H:i:s', time() + 86400) . ' GMT');
    
    // Ausgeben
    imagejpeg($resized, null, 85);
    
    // Aufräumen
    imagedestroy($image);
    imagedestroy($cropped);
    imagedestroy($resized);
    
} catch (Exception $e) {
    http_response_code(500);
    header('Content-Type: text/plain');
    echo 'Fehler: ' . $e->getMessage();
}
