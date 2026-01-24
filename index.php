<?php
/**
 * CAM2 Admin - Schnelle Personen-Benennung (5 neueste Gesichter mit hoher Qualität)
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
        [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
            PDO::ATTR_EMULATE_PREPARES => false
        ]
    );
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
        .person-quick-card {
            display: grid;
            grid-template-columns: 150px 1fr 350px;
            gap: 15px;
            margin-bottom: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #f9f9f9;
        }
        .person-quick-card img.face-crop { width: 150px; height: 150px; object-fit: cover; border-radius: 8px; border: 1px solid #ccc; }
        .person-quick-card img.annotated-preview {
            width: 350px;
            height: auto;
            max-height: 200px;
            object-fit: contain;
            border-radius: 8px;
            border: 2px solid #4caf50;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .person-quick-card img.annotated-preview:hover { transform: scale(1.02); }
        .person-quick-info { display: flex; flex-direction: column; justify-content: center; }
        .stats .unknown { border-bottom: 4px solid #f44336; }
        .stats .known { border-bottom: 4px solid #4caf50; }
        .btn-danger { background-color: #f44336; color: white; border: none; padding: 8px 16px; cursor: pointer; border-radius: 4px; margin-left: 5px; }
        .btn-danger:hover { background-color: #da190b; }

        /* Modal für großes Bild */
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.9); }
        .modal-content { margin: auto; display: block; max-width: 95%; max-height: 95%; margin-top: 2%; }
        .modal-close { position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }
        .modal-close:hover { color: #bbb; }
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
                <?php foreach ($persons as $person):
                    // Annotiertes Bild-Pfad generieren
                    $basename = basename($person['file_path']);
                    $annotated_path = '/annotated/annotated_' . $basename;
                    $annotated_full_path = $_SERVER['DOCUMENT_ROOT'] . $annotated_path;
                    $has_annotated = file_exists($annotated_full_path);
                ?>
                    <div class="person-quick-card">
                        <!-- Gesichts-Crop -->
                        <img class="face-crop" src="api/crop_detection.php?v=2&id=<?= $person['id'] ?>&type=face&size=150" alt="Gesicht">

                        <!-- Info & Actions -->
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

                            <div class="person-quick-actions" style="margin-top: 10px;">
                                <input type="text" id="name_<?= $person['id'] ?>"
                                       placeholder="Name..."
                                       list="nameSuggestions"
                                       onkeypress="if(event.key==='Enter') renamePerson(<?= $person['id'] ?>)">
                                <button class="btn btn-primary" onclick="renamePerson(<?= $person['id'] ?>)">Speichern</button>
                                <button class="btn btn-danger" onclick="deleteFace(<?= $person['id'] ?>)">Löschen</button>
                            </div>
                        </div>

                        <!-- Annotiertes Bild mit Objekt-Boxen -->
                        <div class="annotated-container">
                            <?php if ($has_annotated): ?>
                                <img class="annotated-preview"
                                     src="<?= $annotated_path ?>?v=<?= time() ?>"
                                     alt="Annotiert"
                                     onclick="showFullImage('<?= $annotated_path ?>')"
                                     title="Klicken für Vollansicht">
                                <div style="text-align:center; margin-top:5px; font-size:11px; color:#666;">
                                    📦 Mit Objekt-Boxen (klicken für groß)
                                </div>
                            <?php else: ?>
                                <div style="padding:20px; text-align:center; color:#999; border:1px dashed #ccc; border-radius:8px;">
                                    Kein annotiertes Bild verfügbar<br>
                                    <small>Starten Sie watchdog2.py mit --save-annotated</small>
                                </div>
                            <?php endif; ?>
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

    <!-- Modal für Vollbild-Ansicht -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

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

    async function deleteFace(faceId) {
        try {
            const response = await fetch('api/delete_faces.php', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ face_ids: [faceId] })
            });
            const result = await response.json();
            if (result.success) {
                location.reload();
            } else {
                alert('Fehler: ' + result.error);
            }
        } catch (e) {
            alert('Fehler: ' + e);
        }
    }

    function showFullImage(imagePath) {
        const modal = document.getElementById('imageModal');
        const modalImg = document.getElementById('modalImage');
        modal.style.display = 'block';
        modalImg.src = imagePath;
    }

    function closeModal() {
        document.getElementById('imageModal').style.display = 'none';
    }

    // ESC-Taste schließt Modal
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeModal();
        }
    });
    </script>
</body>
</html>
