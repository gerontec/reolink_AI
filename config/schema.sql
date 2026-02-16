-- Reolink AI Watchdog Database Schema

CREATE TABLE IF NOT EXISTS cam_video_archive (
    id INT PRIMARY KEY AUTO_INCREMENT,
    filename VARCHAR(255) UNIQUE,
    filepath VARCHAR(512),
    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    duration_seconds FLOAT,
    filesize_mb FLOAT,
    trigger_object_type VARCHAR(20),
    trigger_object_id INT,
    frame_width INT,
    frame_height INT,
    fps INT,
    total_detections INT DEFAULT 0,
    archived BOOLEAN DEFAULT FALSE,
    notes VARCHAR(512),
    INDEX idx_recorded_at (recorded_at)
);

CREATE TABLE IF NOT EXISTS cam_detected_objects (
    id INT PRIMARY KEY AUTO_INCREMENT,
    object_hash VARCHAR(64) UNIQUE,
    object_type VARCHAR(20),
    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_detections INT DEFAULT 1,
    times_crossed_line INT DEFAULT 0,
    first_video_id INT,
    INDEX idx_object_type (object_type),
    INDEX idx_object_hash (object_hash),
    FOREIGN KEY (first_video_id) REFERENCES cam_video_archive(id)
);

CREATE TABLE IF NOT EXISTS cam_detections (
    id INT PRIMARY KEY AUTO_INCREMENT,
    object_hash VARCHAR(64),
    video_id INT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    object_type VARCHAR(20),
    yolo_id INT,
    confidence FLOAT,
    bbox_x1 FLOAT,
    bbox_y1 FLOAT,
    bbox_x2 FLOAT,
    bbox_y2 FLOAT,
    in_upper_zone BOOLEAN DEFAULT FALSE,
    crossed_line BOOLEAN DEFAULT FALSE,
    INDEX idx_timestamp (timestamp),
    INDEX idx_object_hash (object_hash),
    FOREIGN KEY (video_id) REFERENCES cam_video_archive(id)
);

CREATE TABLE IF NOT EXISTS cam_scene_stats (
    id INT PRIMARY KEY AUTO_INCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    person_count INT DEFAULT 0,
    car_count INT DEFAULT 0,
    truck_count INT DEFAULT 0,
    motorcycle_count INT DEFAULT 0,
    bus_count INT DEFAULT 0,
    total_objects INT DEFAULT 0,
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE IF NOT EXISTS cam_image_classification (
    id INT PRIMARY KEY AUTO_INCREMENT,
    video_id INT,
    scene_stat_id INT,
    classified_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(64) DEFAULT 'yolov8m-cls',
    img_size INT DEFAULT 224,
    top1_class_id INT,
    top1_class_name VARCHAR(128),
    top1_confidence FLOAT,
    top2_class_name VARCHAR(128),
    top2_confidence FLOAT,
    top3_class_name VARCHAR(128),
    top3_confidence FLOAT,
    top5_confidence_sum FLOAT,
    inference_time_ms FLOAT,
    FOREIGN KEY (video_id) REFERENCES cam_video_archive(id),
    FOREIGN KEY (scene_stat_id) REFERENCES cam_scene_stats(id)
);

CREATE TABLE IF NOT EXISTS cam_face_recognitions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    detection_id INT,
    video_id INT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    recognized BOOLEAN DEFAULT FALSE,
    person_name VARCHAR(128),
    distance FLOAT,
    face_confidence FLOAT,
    face_bbox_x INT,
    face_bbox_y INT,
    face_bbox_w INT,
    face_bbox_h INT,
    INDEX idx_timestamp (timestamp),
    INDEX idx_person_name (person_name),
    FOREIGN KEY (detection_id) REFERENCES cam_detections(id),
    FOREIGN KEY (video_id) REFERENCES cam_video_archive(id)
);

CREATE TABLE IF NOT EXISTS cam_face_embeddings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    person_id VARCHAR(128),
    embedding_vector TEXT,
    face_quality FLOAT,
    bbox_width INT,
    bbox_height INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
    match_count INT DEFAULT 1,
    is_enrolled BOOLEAN DEFAULT FALSE,
    source VARCHAR(64) DEFAULT 'auto',
    notes VARCHAR(512),
    INDEX idx_person_id (person_id)
);

CREATE TABLE IF NOT EXISTS cam_daily_stats (
    id INT PRIMARY KEY AUTO_INCREMENT,
    date DATETIME UNIQUE,
    total_detections INT DEFAULT 0,
    person_count INT DEFAULT 0,
    car_count INT DEFAULT 0,
    motorcycle_count INT DEFAULT 0,
    bus_count INT DEFAULT 0,
    truck_count INT DEFAULT 0,
    line_crossings INT DEFAULT 0,
    upper_zone_movements INT DEFAULT 0,
    videos_recorded INT DEFAULT 0,
    total_video_duration FLOAT DEFAULT 0,
    INDEX idx_date (date)
);
