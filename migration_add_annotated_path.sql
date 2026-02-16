-- Migration: Add annotated_image_path to cam2_recordings
-- Adds support for storing the path to annotated images

USE wagodb;

-- Add column if it doesn't exist
ALTER TABLE cam2_recordings
ADD COLUMN IF NOT EXISTS annotated_image_path VARCHAR(255) DEFAULT NULL
AFTER analyzed;

-- Verify the change
DESCRIBE cam2_recordings;
