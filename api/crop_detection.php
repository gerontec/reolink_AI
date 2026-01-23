<?php
/**
 * Crop detection - mit erweitertem Kopfbereich
 */

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
    error_log("DB Error in crop_detection: " . $e->getMessage());
    http_response_code(500);
    header('Content-Type: text/plain');
    echo "DB Error";
    exit;
}

$id = intval($_GET['id'] ?? 0);
$type = $_GET['type'] ?? 'person';
$size = intval($_GET['size'] ?? 200);
$padding = intval($_GET['padding'] ?? 20);

if ($id <= 0) {
    http_response_code(400);
    header('Content-Type: text/plain');
    echo "Invalid ID";
    exit;
}

$size = min(max($size, 50), 800);

try {
    if ($type === 'face') {
        $sql = "
            SELECT f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2, r.file_path
            FROM cam2_detected_faces f
            JOIN cam2_recordings r ON f.recording_id = r.id
            WHERE f.id = ?
        ";
    } else {
        $sql = "
            SELECT o.bbox_x1, o.bbox_y1, o.bbox_x2, o.bbox_y2, r.file_path
            FROM cam2_detected_objects o
            JOIN cam2_recordings r ON o.recording_id = r.id
            WHERE o.id = ? AND o.object_class = 'person'
        ";
    }
    
    $stmt = $pdo->prepare($sql);
    $stmt->execute([$id]);
    $item = $stmt->fetch();
    
    if (!$item) {
        error_log("No item found for ID: $id, type: $type");
        http_response_code(404);
        header('Content-Type: text/plain');
        echo "Not found in DB";
        exit;
    }
    
    $image_path = '/var/www/web1/' . $item['file_path'];
    
    if (!file_exists($image_path)) {
        error_log("Image file not found: $image_path");
        http_response_code(404);
        header('Content-Type: text/plain');
        echo "Image file not found: " . $item['file_path'];
        exit;
    }
    
    $image = @imagecreatefromjpeg($image_path);
    if (!$image) {
        error_log("Cannot load image: $image_path");
        http_response_code(500);
        header('Content-Type: text/plain');
        echo "Cannot load image";
        exit;
    }
    
    $img_width = imagesx($image);
    $img_height = imagesy($image);
    
    // Original Bounding Box
    $orig_width = $item['bbox_x2'] - $item['bbox_x1'];
    $orig_height = $item['bbox_y2'] - $item['bbox_y1'];
    
    // Erweiterte Box - 20% mehr nach oben für Kopf
    $extra_top = $orig_height * 0.20;  // 20% extra nach oben
    
    // Bounding Box mit asymmetrischem Padding (mehr oben)
    $x1 = max(0, $item['bbox_x1'] - $padding);
    $y1 = max(0, $item['bbox_y1'] - $padding - $extra_top);  // Extra Platz oben
    $x2 = min($img_width, $item['bbox_x2'] + $padding);
    $y2 = min($img_height, $item['bbox_y2'] + $padding);
    
    $crop_width = $x2 - $x1;
    $crop_height = $y2 - $y1;
    
    if ($crop_width <= 0 || $crop_height <= 0) {
        error_log("Invalid crop dimensions: {$crop_width}x{$crop_height}");
        http_response_code(500);
        header('Content-Type: text/plain');
        echo "Invalid dimensions";
        exit;
    }
    
    $cropped = imagecrop($image, [
        'x' => (int)$x1,
        'y' => (int)$y1,
        'width' => (int)$crop_width,
        'height' => (int)$crop_height
    ]);
    
    if (!$cropped) {
        error_log("Crop failed");
        http_response_code(500);
        header('Content-Type: text/plain');
        echo "Crop failed";
        exit;
    }
    
    // Auf gewünschte Größe skalieren
    $resized = imagescale($cropped, $size, -1, IMG_BICUBIC);
    
    // Ausgabe
    header('Content-Type: image/jpeg');
    header('Cache-Control: public, max-age=3600');
    header('Expires: ' . gmdate('D, d M Y H:i:s', time() + 3600) . ' GMT');
    
    imagejpeg($resized, null, 85);
    
    // Aufräumen
    imagedestroy($image);
    imagedestroy($cropped);
    imagedestroy($resized);
    
} catch (Exception $e) {
    error_log("Exception in crop_detection: " . $e->getMessage());
    http_response_code(500);
    header('Content-Type: text/plain');
    echo "Error: " . $e->getMessage();
}
