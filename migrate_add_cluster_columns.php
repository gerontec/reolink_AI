<?php
/**
 * PHP Migration: Add face clustering columns
 * Alternative zu migrate_schema.py
 */

require_once __DIR__ . '/config.php';

try {
    $pdo = getDbConnection();
    echo "✓ Datenbankverbindung erfolgreich\n\n";

    // Check if face_embedding exists
    $stmt = $pdo->query("
        SELECT COUNT(*) as count
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'wagodb'
        AND TABLE_NAME = 'cam2_detected_faces'
        AND COLUMN_NAME = 'face_embedding'
    ");
    $has_embedding = $stmt->fetchColumn() > 0;

    // Check if face_cluster_id exists
    $stmt = $pdo->query("
        SELECT COUNT(*) as count
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'wagodb'
        AND TABLE_NAME = 'cam2_detected_faces'
        AND COLUMN_NAME = 'face_cluster_id'
    ");
    $has_cluster_id = $stmt->fetchColumn() > 0;

    echo "Aktueller Status:\n";
    echo "  face_embedding:   " . ($has_embedding ? '✓ Existiert' : '✗ Fehlt') . "\n";
    echo "  face_cluster_id:  " . ($has_cluster_id ? '✓ Existiert' : '✗ Fehlt') . "\n\n";

    $changes_made = false;

    // Add face_embedding if missing
    if (!$has_embedding) {
        echo "→ Füge face_embedding hinzu...\n";
        $pdo->exec("
            ALTER TABLE cam2_detected_faces
            ADD COLUMN face_embedding BLOB DEFAULT NULL
            AFTER bbox_y2
        ");
        echo "  ✓ face_embedding hinzugefügt\n\n";
        $changes_made = true;
    } else {
        echo "→ face_embedding bereits vorhanden (überspringe)\n\n";
    }

    // Add face_cluster_id if missing
    if (!$has_cluster_id) {
        echo "→ Füge face_cluster_id hinzu...\n";
        $pdo->exec("
            ALTER TABLE cam2_detected_faces
            ADD COLUMN face_cluster_id INT DEFAULT NULL
            AFTER face_embedding
        ");
        echo "  ✓ face_cluster_id hinzugefügt\n\n";
        $changes_made = true;
    } else {
        echo "→ face_cluster_id bereits vorhanden (überspringe)\n\n";
    }

    // Check and add index if missing
    $stmt = $pdo->query("
        SELECT COUNT(*) as count
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = 'wagodb'
        AND TABLE_NAME = 'cam2_detected_faces'
        AND INDEX_NAME = 'idx_cluster'
    ");
    $has_index = $stmt->fetchColumn() > 0;

    if (!$has_index && $has_cluster_id) {
        echo "→ Erstelle Index idx_cluster...\n";
        $pdo->exec("
            ALTER TABLE cam2_detected_faces
            ADD INDEX idx_cluster (face_cluster_id)
        ");
        echo "  ✓ Index idx_cluster erstellt\n\n";
        $changes_made = true;
    } elseif ($has_index) {
        echo "→ Index idx_cluster bereits vorhanden (überspringe)\n\n";
    }

    if ($changes_made) {
        echo "✅ Migration erfolgreich abgeschlossen!\n";
        echo "\nNächste Schritte:\n";
        echo "1. python3 cam2_cluster_faces.py  # Clustering ausführen\n";
        echo "2. Dann index.php im Browser öffnen\n";
    } else {
        echo "✅ Alle Spalten bereits vorhanden - nichts zu tun!\n";
    }

} catch (PDOException $e) {
    echo "❌ FEHLER: " . $e->getMessage() . "\n";
    exit(1);
}
