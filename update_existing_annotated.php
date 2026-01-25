#!/usr/bin/env php
<?php
/**
 * Update Script: Link existing annotated images to recordings
 *
 * Scans the annotated/ directory and links images to their recordings
 */

$db_config = [
    'host' => 'localhost',
    'database' => 'wagodb',
    'user' => 'gh',
    'password' => 'a12345',
    'charset' => 'utf8mb4'
];

$annotated_dir = '/var/www/web1/annotated';

try {
    echo "Connecting to database...\n";
    $pdo = new PDO(
        "mysql:host={$db_config['host']};dbname={$db_config['database']};charset={$db_config['charset']}",
        $db_config['user'],
        $db_config['password'],
        [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]
    );

    echo "Scanning annotated directory: $annotated_dir\n";

    if (!is_dir($annotated_dir)) {
        die("✗ Directory not found: $annotated_dir\n");
    }

    $files = glob("$annotated_dir/annotated_*.jpg");
    echo "Found " . count($files) . " annotated images\n\n";

    $updated = 0;
    $skipped = 0;

    foreach ($files as $file) {
        $filename = basename($file);

        // Extract original filename: annotated_Camera1_00_20260125115843.jpg -> Camera1_00_20260125115843.jpg
        $original_filename = preg_replace('/^annotated_/', '', $filename);

        // Find recording with this filename
        $stmt = $pdo->prepare("
            SELECT id, file_path
            FROM cam2_recordings
            WHERE file_path LIKE ?
        ");
        $stmt->execute(["%$original_filename"]);
        $recording = $stmt->fetch(PDO::FETCH_ASSOC);

        if ($recording) {
            // Update with relative path
            $rel_path = "annotated/$filename";

            $update = $pdo->prepare("
                UPDATE cam2_recordings
                SET annotated_image_path = ?
                WHERE id = ?
            ");
            $update->execute([$rel_path, $recording['id']]);

            echo "✓ Linked: $filename -> Recording #{$recording['id']} ({$recording['file_path']})\n";
            $updated++;
        } else {
            echo "⊘ Skipped: $filename (no matching recording found)\n";
            $skipped++;
        }
    }

    echo "\n========================================\n";
    echo "✓ Updated: $updated\n";
    echo "⊘ Skipped: $skipped\n";
    echo "========================================\n";

} catch (PDOException $e) {
    echo "✗ Error: " . $e->getMessage() . "\n";
    exit(1);
}
