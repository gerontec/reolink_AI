#!/usr/bin/env php
<?php
/**
 * Debug Script: Check annotated images setup
 */

require_once __DIR__ . '/config.php';

echo "=== ANNOTATED IMAGES DEBUG ===\n\n";

try {
    $pdo = getDbConnection();

    // 1. Check if column exists
    echo "1. Checking database schema...\n";
    $stmt = $pdo->query("SHOW COLUMNS FROM cam2_recordings LIKE 'annotated_image_path'");
    $column = $stmt->fetch();

    if ($column) {
        echo "   ✓ Column 'annotated_image_path' exists\n\n";
    } else {
        echo "   ✗ Column 'annotated_image_path' NOT FOUND\n";
        echo "   → Run: php migrate_db.php\n\n";
        exit(1);
    }

    // 2. Check recordings with annotated images
    echo "2. Checking recordings with annotated images...\n";
    $stmt = $pdo->query("
        SELECT COUNT(*) as total,
               SUM(CASE WHEN annotated_image_path IS NOT NULL THEN 1 ELSE 0 END) as with_annotated
        FROM cam2_recordings
    ");
    $stats = $stmt->fetch();

    echo "   Total recordings: {$stats['total']}\n";
    echo "   With annotated images: {$stats['with_annotated']}\n\n";

    if ($stats['with_annotated'] == 0) {
        echo "   ⚠️  No recordings have annotated images!\n";
        echo "   → Run watchdog2.py with --save-annotated\n";
        echo "   → Or run: php update_existing_annotated.php\n\n";
    }

    // 3. Show sample recordings with faces
    echo "3. Sample faces with annotated images:\n";
    $stmt = $pdo->query("
        SELECT
            f.id as face_id,
            f.confidence,
            r.file_path,
            r.annotated_image_path,
            r.camera_name
        FROM cam2_detected_faces f
        JOIN cam2_recordings r ON f.recording_id = r.id
        ORDER BY f.detected_at DESC
        LIMIT 5
    ");

    while ($row = $stmt->fetch()) {
        echo "\n   Face #{$row['face_id']}:\n";
        echo "     Camera: {$row['camera_name']}\n";
        echo "     Confidence: " . round($row['confidence'] * 100, 1) . "%\n";
        echo "     Original: {$row['file_path']}\n";

        if ($row['annotated_image_path']) {
            echo "     Annotated: {$row['annotated_image_path']}\n";

            // Check if file exists
            $full_path = '/var/www/web1/' . $row['annotated_image_path'];
            if (file_exists($full_path)) {
                $size = filesize($full_path);
                echo "     ✓ File exists (" . round($size/1024, 1) . " KB)\n";
            } else {
                echo "     ✗ File NOT FOUND: $full_path\n";
            }
        } else {
            echo "     ⚠️  No annotated image path\n";
        }
    }

    echo "\n\n4. Checking /var/www/web1/annotated/ directory...\n";
    $annotated_dir = '/var/www/web1/annotated';

    if (is_dir($annotated_dir)) {
        $files = glob("$annotated_dir/annotated_*.jpg");
        echo "   ✓ Directory exists\n";
        echo "   Found " . count($files) . " annotated images\n";

        if (count($files) > 0) {
            echo "\n   Sample files:\n";
            foreach (array_slice($files, 0, 3) as $file) {
                echo "     - " . basename($file) . " (" . round(filesize($file)/1024, 1) . " KB)\n";
            }
        }
    } else {
        echo "   ✗ Directory NOT FOUND: $annotated_dir\n";
        echo "   → watchdog2.py with --save-annotated will create it\n";
    }

    echo "\n=== DEBUG COMPLETE ===\n";

} catch (PDOException $e) {
    echo "✗ Database error: " . $e->getMessage() . "\n";
    exit(1);
}
