-- Migration: Add face clustering columns to cam2_detected_faces
-- Date: 2026-02-17
-- Purpose: Enable face clustering for grouping identical unknown faces

USE wagodb;

-- Add face_embedding column (512-dim ArcFace vectors as BLOB)
ALTER TABLE cam2_detected_faces
ADD COLUMN face_embedding BLOB DEFAULT NULL
AFTER bbox_y2;

-- Add face_cluster_id column (cluster ID assigned by DBSCAN)
ALTER TABLE cam2_detected_faces
ADD COLUMN face_cluster_id INT DEFAULT NULL
AFTER face_embedding;

-- Add index for fast cluster queries
ALTER TABLE cam2_detected_faces
ADD INDEX idx_cluster (face_cluster_id);

-- Verify changes
SHOW COLUMNS FROM cam2_detected_faces;
