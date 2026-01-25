#!/usr/bin/env php
<?php
/**
 * Database Migration: Add annotated_image_path column to cam2_recordings
 */

require_once __DIR__ . '/config.php';

try {
    echo "Connecting to database...\n";
    $pdo = getDbConnection();

    // Check if column exists
    $stmt = $pdo->query("SHOW COLUMNS FROM cam2_recordings LIKE 'annotated_image_path'");
    $result = $stmt->fetch();

    if ($result) {
        echo "✓ Column annotated_image_path already exists\n";
    } else {
        echo "Adding column annotated_image_path...\n";
        $pdo->exec("
            ALTER TABLE cam2_recordings
            ADD COLUMN annotated_image_path VARCHAR(255) DEFAULT NULL
            AFTER analyzed
        ");
        echo "✓ Column added successfully\n";
    }

    // Verify
    echo "\nTable structure:\n";
    $stmt = $pdo->query("DESCRIBE cam2_recordings");
    while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
        echo "  {$row['Field']}: {$row['Type']}\n";
    }

    echo "\n✓ Migration completed successfully\n";

} catch (PDOException $e) {
    echo "✗ Error: " . $e->getMessage() . "\n";
    exit(1);
}
