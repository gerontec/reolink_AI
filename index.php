<?php
/**
 * CAM2 Admin - Schnelle Personen-Benennung (5 neueste Gesichter mit hoher Qualität)
 */

require_once __DIR__ . '/config.php';

try {
    $pdo = getDbConnection();
} catch (PDOException $e) {
    die("DB-Fehler: " . htmlspecialchars($e->getMessage()));
}

// --- Statistiken (Neue Tabellenstruktur) ---
$stats = [
    'total_persons' => $pdo->query("SELECT COUNT(*) FROM cam2_detected_faces")->fetchColumn(),
    'unnamed_persons' => $pdo->query("SELECT COUNT(*) FROM cam2_detected_faces WHERE person_name = 'Unknown'")->fetchColumn(),
    'named_persons' => $pdo->query("SELECT COUNT(DISTINCT person_name) FROM cam2_detected_faces WHERE person_name != 'Unknown'")->fetchColumn(),
    'total_recordings' => $pdo->query("SELECT COUNT(*) FROM cam2_recordings")->fetchColumn(),
];

// Filter
$show_all = isset($_GET['all']);
$min_confidence = floatval($_GET['min_conf'] ?? 0.4); // Standard etwas höher für Gesichter
$min_size = intval($_GET['min_size'] ?? 40);       // Gesichter sind oft kleiner als ganze Körper
$limit = intval($_GET['limit'] ?? 5);

// --- Personen/Gesichter abrufen (5 neueste mit hoher Qualität) ---
$sql = "
    SELECT
        f.id,
        f.confidence,
        f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
        f.person_name,
        f.detected_at,
        r.file_path,
        r.camera_name,
        r.recorded_at,
        r.annotated_image_path,
        (f.bbox_x2 - f.bbox_x1) * (f.bbox_y2 - f.bbox_y1) as area,
        (f.bbox_x2 - f.bbox_x1) as width,
        (f.bbox_y2 - f.bbox_y1) as height
    FROM cam2_detected_faces f
    JOIN cam2_recordings r ON f.recording_id = r.id
    WHERE f.confidence >= ?
      AND (f.bbox_x2 - f.bbox_x1) >= ?
";

if (!$show_all) {
    $sql .= " AND f.person_name = 'Unknown'";
}

$sql .= " ORDER BY f.detected_at DESC, f.confidence DESC LIMIT ?";

$stmt = $pdo->prepare($sql);
$stmt->execute([$min_confidence, $min_size, $limit]);
$persons = $stmt->fetchAll();

// Vorschläge für Autocomplete (Alle bereits vergebenen Namen)
$named = $pdo->query("
    SELECT person_name, COUNT(*) as count
    FROM cam2_detected_faces
    WHERE person_name != 'Unknown'
    GROUP BY person_name
    ORDER BY person_name
")->fetchAll();
?>
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <title>CAM Admin - Gesichtserkennung</title>
    <link rel="stylesheet" href="css/style.css">
    <style>
        /* Bestehende Styles beibehalten... */
        .person-quick-card img { width: 150px; height: 150px; object-fit: cover; border-radius: 8px; border: 1px solid #ccc; }
        .stats .unknown { border-bottom: 4px solid #f44336; }
        .stats .known { border-bottom: 4px solid #4caf50; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>👤 CAM2 Admin - Gesichter zuordnen (5 neueste)</h1>
            <div class="stats">
                <div class="stat-box">
                    <span class="stat-value"><?= number_format($stats['total_persons']) ?></span>
                    <span class="stat-label">Gesichter Total</span>
                </div>
                <div class="stat-box unknown">
                    <span class="stat-value"><?= number_format($stats['unnamed_persons']) ?></span>
                    <span class="stat-label">Unbekannt</span>
                </div>
                <div class="stat-box known">
                    <span class="stat-value"><?= $stats['named_persons'] ?></span>
                    <span class="stat-label">Personen</span>
                </div>
            </div>
        </header>

        <div class="filters">
            <form method="GET">
                <label>Anzahl: <input type="number" name="limit" value="<?= $limit ?>" style="width:60px;"></label>
                <label>Min. Conf: <input type="number" step="0.1" name="min_conf" value="<?= $min_confidence ?>" style="width:60px;"></label>
                <label><input type="checkbox" name="all" <?= $show_all ? 'checked' : '' ?>> Auch benannte</label>
                <button type="submit" class="btn">Aktualisieren</button>
            </form>
        </div>

        <?php if (empty($persons)): ?>
            <div class="no-results"><h2>Keine unbenannten Gesichter gefunden.</h2></div>
        <?php else: ?>
            <div class="quick-rename">
                <?php foreach ($persons as $person): ?>
                    <div class="person-quick-card">
                        <?php if (!empty($person['annotated_image_path'])): ?>
                            <img src="/web1/<?= htmlspecialchars($person['annotated_image_path']) ?>" alt="Annotiertes Bild" style="max-width: 300px; height: auto;">
                            <div style="font-size: 0.8em; color: #666; margin-top: 5px;">📸 Mit YOLO Detektionen</div>
                        <?php else: ?>
                            <img src="api/crop_detection.php?v=2&id=<?= $person['id'] ?>&type=face&size=150" alt="Gesicht">
                            <div style="font-size: 0.8em; color: #999; margin-top: 5px;">⚠️ Nur Gesichts-Crop</div>
                        <?php endif; ?>

                        <div class="person-quick-info">
                            <h4>Gesicht #<?= $person['id'] ?></h4>
                            <p>
                                <strong>Kamera:</strong> <?= htmlspecialchars($person['camera_name']) ?><br>
                                <strong>Zeit:</strong> <?= $person['recorded_at'] ?><br>
                                <strong>Konfidenz:</strong> <?= round($person['confidence']*100, 1) ?>%
                            </p>
                            <?php if ($person['person_name'] !== 'Unknown'): ?>
                                <p style="color:green">✓ Aktuell: <?= htmlspecialchars($person['person_name']) ?></p>
                            <?php endif; ?>
                        </div>

                        <div class="person-quick-actions">
                            <input type="text" id="name_<?= $person['id'] ?>"
                                   placeholder="Name..."
                                   list="nameSuggestions"
                                   onkeypress="if(event.key==='Enter') renamePerson(<?= $person['id'] ?>)">
                            <button class="btn btn-primary" onclick="renamePerson(<?= $person['id'] ?>)">Speichern</button>
                            <button class="btn btn-danger" onclick="deletePerson(<?= $person['id'] ?>)" style="background-color: #f44336;">🗑️ Löschen</button>
                        </div>
                    </div>
                <?php endforeach; ?>
            </div>
        <?php endif; ?>
    </div>

    <datalist id="nameSuggestions">
        <?php foreach ($named as $n): ?>
            <option value="<?= htmlspecialchars($n['person_name']) ?>">
        <?php endforeach; ?>
    </datalist>

    <script>
    async function renamePerson(faceId) {
        const name = document.getElementById('name_' + faceId).value.trim();
        if (!name) return;

        try {
            const response = await fetch('api/rename_person.php', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ face_id: faceId, person_name: name })
            });
            const result = await response.json();
            if (result.success) location.reload();
            else alert(result.error);
        } catch (e) { alert(e); }
    }

    async function deletePerson(faceId) {
        try {
            const response = await fetch('api/delete_faces.php', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ face_ids: [faceId] })
            });
            const result = await response.json();
            if (result.success) location.reload();
            else alert(result.error);
        } catch (e) { alert(e); }
    }
    </script>
</body>
</html>
