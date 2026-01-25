<?php
/**
 * YOLO Detektionen - Übersicht aller annotierten Bilder
 * Zeigt ALLE Recordings mit YOLO-Detektionen (Personen, Fahrzeuge, Objekte)
 */

require_once __DIR__ . '/config.php';

try {
    $pdo = getDbConnection();
} catch (PDOException $e) {
    die("DB-Fehler: " . htmlspecialchars($e->getMessage()));
}

// Filter
$camera_filter = $_GET['camera'] ?? '';
$date_filter = $_GET['date'] ?? date('Y-m-d');
$object_type = $_GET['type'] ?? '';
$limit = intval($_GET['limit'] ?? 50);

// Statistiken
$stats_sql = "
    SELECT
        COUNT(DISTINCT r.id) as total_annotated,
        COUNT(DISTINCT CASE WHEN o.object_class = 'person' THEN r.id END) as with_persons,
        COUNT(DISTINCT CASE WHEN o.object_class IN ('car','truck','bus','motorcycle') THEN r.id END) as with_vehicles,
        SUM(CASE WHEN o.object_class = 'person' THEN 1 ELSE 0 END) as total_persons,
        SUM(CASE WHEN o.object_class IN ('car','truck','bus','motorcycle') THEN 1 ELSE 0 END) as total_vehicles
    FROM cam2_recordings r
    LEFT JOIN cam2_detected_objects o ON r.id = o.recording_id
    WHERE r.annotated_image_path IS NOT NULL
";
$stats = $pdo->query($stats_sql)->fetch();

// Haupt-Query: Alle Recordings mit annotierten Bildern
$sql = "
    SELECT
        r.id,
        r.camera_name,
        r.file_path,
        r.file_type,
        r.annotated_image_path,
        r.recorded_at,
        COUNT(DISTINCT f.id) as face_count,
        COUNT(DISTINCT CASE WHEN o.object_class = 'person' THEN o.id END) as person_count,
        COUNT(DISTINCT CASE WHEN o.object_class IN ('car','truck','bus','motorcycle') THEN o.id END) as vehicle_count,
        GROUP_CONCAT(DISTINCT o.object_class ORDER BY o.object_class SEPARATOR ', ') as detected_objects
    FROM cam2_recordings r
    LEFT JOIN cam2_detected_faces f ON r.id = f.recording_id
    LEFT JOIN cam2_detected_objects o ON r.id = o.recording_id
    WHERE r.annotated_image_path IS NOT NULL
";

$params = [];

if ($camera_filter) {
    $sql .= " AND r.camera_name = ?";
    $params[] = $camera_filter;
}

if ($date_filter) {
    $sql .= " AND DATE(r.recorded_at) = ?";
    $params[] = $date_filter;
}

if ($object_type) {
    $sql .= " AND EXISTS (
        SELECT 1 FROM cam2_detected_objects o2
        WHERE o2.recording_id = r.id AND o2.object_class = ?
    )";
    $params[] = $object_type;
}

$sql .= " GROUP BY r.id ORDER BY r.recorded_at DESC LIMIT ?";
$params[] = $limit;

$stmt = $pdo->prepare($sql);
$stmt->execute($params);
$recordings = $stmt->fetchAll();

// Kameras für Filter
$cameras = $pdo->query("SELECT DISTINCT camera_name FROM cam2_recordings ORDER BY camera_name")->fetchAll(PDO::FETCH_COLUMN);

// Objekt-Typen für Filter
$object_types = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'dog', 'cat'];
?>
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <title>YOLO Detektionen - Übersicht</title>
    <link rel="stylesheet" href="css/style.css">
    <style>
        .detection-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .detection-card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .detection-card img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .detection-card img:hover {
            transform: scale(1.02);
        }
        .detection-meta {
            margin-top: 10px;
            font-size: 0.9em;
        }
        .detection-meta strong {
            color: #333;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin: 2px;
            font-weight: bold;
        }
        .badge-person { background: #4caf50; color: white; }
        .badge-vehicle { background: #ff9800; color: white; }
        .badge-face { background: #2196f3; color: white; }
        .filter-bar {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }
        .filter-bar select, .filter-bar input {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .video-badge {
            background: #9c27b0;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.75em;
            margin-left: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 YOLO Detektionen - Alle annotierten Bilder</h1>
            <div style="text-align: center; margin: 10px 0;">
                <a href="index.php" style="color: #2196f3; text-decoration: none;">← Zurück zur Gesichtserkennung</a>
            </div>

            <div class="stats">
                <div class="stat-box">
                    <span class="stat-value"><?= number_format($stats['total_annotated']) ?></span>
                    <span class="stat-label">Annotierte Bilder</span>
                </div>
                <div class="stat-box">
                    <span class="stat-value"><?= number_format($stats['with_persons']) ?></span>
                    <span class="stat-label">Mit Personen</span>
                </div>
                <div class="stat-box">
                    <span class="stat-value"><?= number_format($stats['with_vehicles']) ?></span>
                    <span class="stat-label">Mit Fahrzeugen</span>
                </div>
                <div class="stat-box">
                    <span class="stat-value"><?= number_format($stats['total_persons']) ?></span>
                    <span class="stat-label">Personen Total</span>
                </div>
                <div class="stat-box">
                    <span class="stat-value"><?= number_format($stats['total_vehicles']) ?></span>
                    <span class="stat-label">Fahrzeuge Total</span>
                </div>
            </div>
        </header>

        <!-- Filter -->
        <form method="get" class="filter-bar">
            <label>
                Kamera:
                <select name="camera">
                    <option value="">Alle</option>
                    <?php foreach ($cameras as $cam): ?>
                        <option value="<?= htmlspecialchars($cam) ?>" <?= $cam === $camera_filter ? 'selected' : '' ?>>
                            <?= htmlspecialchars($cam) ?>
                        </option>
                    <?php endforeach; ?>
                </select>
            </label>

            <label>
                Datum:
                <input type="date" name="date" value="<?= htmlspecialchars($date_filter) ?>">
            </label>

            <label>
                Objekt-Typ:
                <select name="type">
                    <option value="">Alle</option>
                    <?php foreach ($object_types as $type): ?>
                        <option value="<?= $type ?>" <?= $type === $object_type ? 'selected' : '' ?>>
                            <?= ucfirst($type) ?>
                        </option>
                    <?php endforeach; ?>
                </select>
            </label>

            <label>
                Limit:
                <select name="limit">
                    <option value="20" <?= $limit === 20 ? 'selected' : '' ?>>20</option>
                    <option value="50" <?= $limit === 50 ? 'selected' : '' ?>>50</option>
                    <option value="100" <?= $limit === 100 ? 'selected' : '' ?>>100</option>
                    <option value="200" <?= $limit === 200 ? 'selected' : '' ?>>200</option>
                </select>
            </label>

            <button type="submit" style="padding: 8px 20px; background: #2196f3; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Filtern
            </button>
        </form>

        <!-- Detektionen Grid -->
        <?php if (empty($recordings)): ?>
            <div class="no-results">
                <h2>Keine annotierten Bilder gefunden</h2>
                <p>Versuche andere Filter-Einstellungen</p>
            </div>
        <?php else: ?>
            <div class="detection-grid">
                <?php foreach ($recordings as $rec): ?>
                    <div class="detection-card">
                        <a href="/web1/<?= htmlspecialchars($rec['annotated_image_path']) ?>" target="_blank">
                            <img src="/web1/<?= htmlspecialchars($rec['annotated_image_path']) ?>"
                                 alt="YOLO Detection"
                                 title="Klicken für Vollbild">
                        </a>

                        <div class="detection-meta">
                            <strong><?= htmlspecialchars($rec['camera_name']) ?></strong>
                            <?php if ($rec['file_type'] === 'mp4'): ?>
                                <span class="video-badge">VIDEO</span>
                            <?php endif; ?>
                            <br>
                            <small><?= date('d.m.Y H:i:s', strtotime($rec['recorded_at'])) ?></small>

                            <div style="margin-top: 8px;">
                                <?php if ($rec['person_count'] > 0): ?>
                                    <span class="badge badge-person">👤 <?= $rec['person_count'] ?> Person(en)</span>
                                <?php endif; ?>

                                <?php if ($rec['vehicle_count'] > 0): ?>
                                    <span class="badge badge-vehicle">🚗 <?= $rec['vehicle_count'] ?> Fahrzeug(e)</span>
                                <?php endif; ?>

                                <?php if ($rec['face_count'] > 0): ?>
                                    <span class="badge badge-face">😊 <?= $rec['face_count'] ?> Gesicht(er)</span>
                                <?php endif; ?>
                            </div>

                            <?php if ($rec['detected_objects']): ?>
                                <div style="margin-top: 5px; font-size: 0.8em; color: #666;">
                                    Objekte: <?= htmlspecialchars($rec['detected_objects']) ?>
                                </div>
                            <?php endif; ?>
                        </div>
                    </div>
                <?php endforeach; ?>
            </div>
        <?php endif; ?>
    </div>
</body>
</html>
