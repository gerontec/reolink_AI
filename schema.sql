/*M!999999\- enable the sandbox mode */ 
-- MariaDB dump 10.19-11.8.3-MariaDB, for debian-linux-gnu (x86_64)
--
-- Host: kellertreppe.fritz.box    Database: wagodb
-- ------------------------------------------------------
-- Server version	10.11.14-MariaDB-0+deb12u2

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*M!100616 SET @OLD_NOTE_VERBOSITY=@@NOTE_VERBOSITY, NOTE_VERBOSITY=0 */;

--
-- Table structure for table `K2020`
--

DROP TABLE IF EXISTS `K2020`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `K2020` (
  `Kode1` varchar(6) DEFAULT NULL,
  `Hauptdiagnose` varchar(179) DEFAULT NULL,
  `Faelle` int(6) DEFAULT NULL,
  `Prozent` varchar(5) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `K2020HJ2`
--

DROP TABLE IF EXISTS `K2020HJ2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `K2020HJ2` (
  `Kode` varchar(6) DEFAULT NULL,
  `Hauptdiagnose` varchar(179) DEFAULT NULL,
  `Faelle` varchar(6) DEFAULT NULL,
  `Prozent` varchar(20) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `K2021`
--

DROP TABLE IF EXISTS `K2021`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `K2021` (
  `Kode` varchar(6) DEFAULT NULL,
  `Hauptdiagnose` varchar(179) DEFAULT NULL,
  `Faelle` int(6) DEFAULT NULL,
  `Prozent` varchar(5) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `K2021HJ2`
--

DROP TABLE IF EXISTS `K2021HJ2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `K2021HJ2` (
  `Kode` varchar(6) DEFAULT NULL,
  `Hauptdiagnose` varchar(179) DEFAULT NULL,
  `Faelle` varchar(7) DEFAULT NULL,
  `vh` varchar(5) DEFAULT NULL,
  `Prozent` decimal(3,2) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `blocks`
--

DROP TABLE IF EXISTS `blocks`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `blocks` (
  `number` bigint(20) NOT NULL,
  `hash` varchar(64) NOT NULL,
  `parent_hash` varchar(64) NOT NULL,
  `nonce` varchar(64) DEFAULT NULL,
  `sha3_uncles` varchar(64) DEFAULT NULL,
  `logs_bloom` longtext DEFAULT NULL,
  `transactions_root` varchar(64) DEFAULT NULL,
  `state_root` varchar(64) DEFAULT NULL,
  `receipts_root` varchar(64) DEFAULT NULL,
  `miner` varchar(42) DEFAULT NULL,
  `difficulty` decimal(38,0) DEFAULT NULL,
  `total_difficulty` decimal(38,0) DEFAULT NULL,
  `size` bigint(20) DEFAULT NULL,
  `extra_data` longtext DEFAULT NULL,
  `gas_limit` bigint(20) DEFAULT NULL,
  `gas_used` bigint(20) DEFAULT NULL,
  `timestamp` bigint(20) DEFAULT NULL,
  `transaction_count` bigint(20) DEFAULT NULL,
  `base_fee_per_gas` decimal(38,0) DEFAULT NULL,
  `withdrawals_root` varchar(64) DEFAULT NULL,
  PRIMARY KEY (`number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_analysis_summary`
--

DROP TABLE IF EXISTS `cam2_analysis_summary`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_analysis_summary` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `recording_id` int(11) NOT NULL,
  `total_faces` int(11) DEFAULT 0,
  `total_objects` int(11) DEFAULT 0,
  `total_vehicles` int(11) DEFAULT 0,
  `max_persons` int(11) DEFAULT 0,
  `scene_category` varchar(50) DEFAULT 'unknown',
  `gpu_used` tinyint(1) DEFAULT 0,
  `analyzed_at` datetime NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `recording_id` (`recording_id`),
  KEY `idx_recording` (`recording_id`),
  KEY `idx_scene` (`scene_category`),
  CONSTRAINT `cam2_analysis_summary_ibfk_1` FOREIGN KEY (`recording_id`) REFERENCES `cam2_recordings` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=33 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_daily_stats`
--

DROP TABLE IF EXISTS `cam2_daily_stats`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_daily_stats` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `date` datetime DEFAULT NULL,
  `total_detections` int(11) DEFAULT NULL,
  `person_count` int(11) DEFAULT NULL,
  `car_count` int(11) DEFAULT NULL,
  `motorcycle_count` int(11) DEFAULT NULL,
  `bus_count` int(11) DEFAULT NULL,
  `truck_count` int(11) DEFAULT NULL,
  `line_crossings` int(11) DEFAULT NULL,
  `upper_zone_movements` int(11) DEFAULT NULL,
  `videos_recorded` int(11) DEFAULT NULL,
  `total_video_duration` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `ix_cam_daily_stats_date` (`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_detected_faces`
--

DROP TABLE IF EXISTS `cam2_detected_faces`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_detected_faces` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `recording_id` int(11) NOT NULL,
  `person_name` varchar(100) NOT NULL,
  `confidence` float NOT NULL,
  `bbox_x1` int(11) DEFAULT NULL,
  `bbox_y1` int(11) DEFAULT NULL,
  `bbox_x2` int(11) DEFAULT NULL,
  `bbox_y2` int(11) DEFAULT NULL,
  `detected_at` timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_person` (`person_name`),
  KEY `idx_recording` (`recording_id`),
  CONSTRAINT `cam2_detected_faces_ibfk_1` FOREIGN KEY (`recording_id`) REFERENCES `cam2_recordings` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_detected_objects`
--

DROP TABLE IF EXISTS `cam2_detected_objects`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_detected_objects` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `recording_id` int(11) NOT NULL,
  `object_class` varchar(50) NOT NULL,
  `confidence` float NOT NULL,
  `bbox_x1` int(11) DEFAULT NULL,
  `bbox_y1` int(11) DEFAULT NULL,
  `bbox_x2` int(11) DEFAULT NULL,
  `bbox_y2` int(11) DEFAULT NULL,
  `parking_spot_id` int(11) DEFAULT NULL,
  `detected_at` timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_class` (`object_class`),
  KEY `idx_recording` (`recording_id`),
  KEY `idx_parking_spot` (`parking_spot_id`),
  CONSTRAINT `cam2_detected_objects_ibfk_1` FOREIGN KEY (`recording_id`) REFERENCES `cam2_recordings` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=41 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_detections`
--

DROP TABLE IF EXISTS `cam2_detections`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_detections` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `object_hash` varchar(64) DEFAULT NULL,
  `video_id` int(11) DEFAULT NULL,
  `timestamp` datetime DEFAULT NULL,
  `object_type` varchar(20) DEFAULT NULL,
  `yolo_id` int(11) DEFAULT NULL,
  `confidence` float DEFAULT NULL,
  `bbox_x1` float DEFAULT NULL,
  `bbox_y1` float DEFAULT NULL,
  `bbox_x2` float DEFAULT NULL,
  `bbox_y2` float DEFAULT NULL,
  `in_upper_zone` tinyint(1) DEFAULT NULL,
  `crossed_line` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `video_id` (`video_id`),
  KEY `ix_cam_detections_timestamp` (`timestamp`),
  KEY `ix_cam_detections_object_hash` (`object_hash`),
  CONSTRAINT `cam2_detections_ibfk_1` FOREIGN KEY (`video_id`) REFERENCES `cam2_video_archive` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=150743 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_face_embeddings`
--

DROP TABLE IF EXISTS `cam2_face_embeddings`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_face_embeddings` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `person_id` varchar(128) DEFAULT NULL,
  `embedding_vector` text DEFAULT NULL,
  `face_quality` float DEFAULT NULL,
  `bbox_width` int(11) DEFAULT NULL,
  `bbox_height` int(11) DEFAULT NULL,
  `created_at` datetime DEFAULT NULL,
  `last_seen` datetime DEFAULT NULL,
  `match_count` int(11) DEFAULT NULL,
  `is_enrolled` tinyint(1) DEFAULT NULL,
  `source` varchar(64) DEFAULT NULL,
  `notes` varchar(512) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `ix_cam_face_embeddings_person_id` (`person_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_face_embeddings2`
--

DROP TABLE IF EXISTS `cam2_face_embeddings2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_face_embeddings2` (
  `FACE_ID` int(11) DEFAULT NULL,
  `DIMENSION` int(11) DEFAULT NULL,
  `VALUE` decimal(35,30) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_face_embeddings_old`
--

DROP TABLE IF EXISTS `cam2_face_embeddings_old`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_face_embeddings_old` (
  `FACE_ID` int(11) DEFAULT NULL,
  `DIMENSION` int(11) DEFAULT NULL,
  `VALUE` decimal(35,30) DEFAULT NULL,
  KEY `idx_faceid` (`FACE_ID`),
  KEY `idx_dim` (`FACE_ID`,`DIMENSION`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_face_meta`
--

DROP TABLE IF EXISTS `cam2_face_meta`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_face_meta` (
  `ID` int(11) NOT NULL,
  `IMG_NAME` varchar(55) DEFAULT NULL,
  `EMBEDDING` blob DEFAULT NULL,
  PRIMARY KEY (`ID`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_face_recognitions`
--

DROP TABLE IF EXISTS `cam2_face_recognitions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_face_recognitions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `detection_id` int(11) DEFAULT NULL,
  `video_id` int(11) DEFAULT NULL,
  `timestamp` datetime DEFAULT NULL,
  `recognized` tinyint(1) DEFAULT NULL,
  `person_name` varchar(128) DEFAULT NULL,
  `distance` float DEFAULT NULL,
  `face_confidence` float DEFAULT NULL,
  `face_bbox_x` int(11) DEFAULT NULL,
  `face_bbox_y` int(11) DEFAULT NULL,
  `face_bbox_w` int(11) DEFAULT NULL,
  `face_bbox_h` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `detection_id` (`detection_id`),
  KEY `video_id` (`video_id`),
  KEY `ix_cam_face_recognitions_timestamp` (`timestamp`),
  KEY `ix_cam_face_recognitions_person_name` (`person_name`),
  CONSTRAINT `cam2_face_recognitions_ibfk_1` FOREIGN KEY (`detection_id`) REFERENCES `cam2_detections` (`id`),
  CONSTRAINT `cam2_face_recognitions_ibfk_2` FOREIGN KEY (`video_id`) REFERENCES `cam2_video_archive` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_image_classification`
--

DROP TABLE IF EXISTS `cam2_image_classification`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_image_classification` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `video_id` int(11) DEFAULT NULL,
  `scene_stat_id` int(11) DEFAULT NULL,
  `classified_at` datetime NOT NULL DEFAULT current_timestamp(),
  `source_frame_timestamp` datetime DEFAULT NULL,
  `image_hash` varchar(64) DEFAULT NULL,
  `model_name` varchar(64) NOT NULL DEFAULT 'yolov8m-cls',
  `model_version` varchar(32) DEFAULT NULL,
  `img_size` int(11) NOT NULL DEFAULT 224,
  `top1_class_id` smallint(5) unsigned NOT NULL,
  `top1_class_name` varchar(128) NOT NULL,
  `top1_confidence` float NOT NULL,
  `top2_class_name` varchar(128) DEFAULT NULL,
  `top2_confidence` float DEFAULT NULL,
  `top3_class_name` varchar(128) DEFAULT NULL,
  `top3_confidence` float DEFAULT NULL,
  `top5_confidence_sum` float DEFAULT NULL,
  `inference_time_ms` float DEFAULT NULL,
  `is_anomaly` tinyint(1) DEFAULT 0,
  PRIMARY KEY (`id`),
  KEY `video_id` (`video_id`),
  KEY `scene_stat_id` (`scene_stat_id`),
  KEY `idx_classified_at` (`classified_at`),
  KEY `idx_top1_class` (`top1_class_name`),
  CONSTRAINT `cam2_image_classification_ibfk_1` FOREIGN KEY (`video_id`) REFERENCES `cam2_video_archive` (`id`) ON DELETE SET NULL,
  CONSTRAINT `cam2_image_classification_ibfk_2` FOREIGN KEY (`scene_stat_id`) REFERENCES `cam2_scene_stats` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB AUTO_INCREMENT=307 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_parking_stats`
--

DROP TABLE IF EXISTS `cam2_parking_stats`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_parking_stats` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `parking_spot_id` int(11) NOT NULL,
  `date` date NOT NULL,
  `hour` int(11) NOT NULL,
  `occupancy_count` int(11) DEFAULT 0,
  `vehicle_type` varchar(50) DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `unique_spot_time` (`parking_spot_id`,`date`,`hour`,`vehicle_type`),
  KEY `idx_spot` (`parking_spot_id`),
  KEY `idx_date` (`date`),
  KEY `idx_spot_date` (`parking_spot_id`,`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_recordings`
--

DROP TABLE IF EXISTS `cam2_recordings`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_recordings` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `camera_name` varchar(50) NOT NULL,
  `file_path` varchar(255) NOT NULL,
  `file_type` enum('jpg','mp4') NOT NULL,
  `file_size` bigint(20) NOT NULL,
  `recorded_at` datetime NOT NULL,
  `analyzed` tinyint(1) DEFAULT 0,
  `created_at` datetime NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `file_path` (`file_path`),
  KEY `idx_camera` (`camera_name`),
  KEY `idx_recorded` (`recorded_at`),
  KEY `idx_analyzed` (`analyzed`),
  KEY `idx_type` (`file_type`)
) ENGINE=InnoDB AUTO_INCREMENT=32 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_scene_stats`
--

DROP TABLE IF EXISTS `cam2_scene_stats`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_scene_stats` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `timestamp` datetime DEFAULT current_timestamp(),
  `person_count` int(11) DEFAULT 0,
  `car_count` int(11) DEFAULT 0,
  `truck_count` int(11) DEFAULT 0,
  `motorcycle_count` int(11) DEFAULT 0,
  `bus_count` int(11) DEFAULT 0,
  `total_objects` int(11) DEFAULT 0,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=17566 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cam2_video_archive`
--

DROP TABLE IF EXISTS `cam2_video_archive`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cam2_video_archive` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `filename` varchar(255) DEFAULT NULL,
  `filepath` varchar(512) DEFAULT NULL,
  `recorded_at` datetime DEFAULT NULL,
  `duration_seconds` float DEFAULT NULL,
  `filesize_mb` float DEFAULT NULL,
  `trigger_object_type` varchar(20) DEFAULT NULL,
  `trigger_object_id` int(11) DEFAULT NULL,
  `frame_width` int(11) DEFAULT NULL,
  `frame_height` int(11) DEFAULT NULL,
  `fps` int(11) DEFAULT NULL,
  `total_detections` int(11) DEFAULT NULL,
  `archived` tinyint(1) DEFAULT NULL,
  `notes` varchar(512) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `ix_cam_video_archive_filename` (`filename`),
  KEY `ix_cam_video_archive_recorded_at` (`recorded_at`)
) ENGINE=InnoDB AUTO_INCREMENT=56 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `category`
--

DROP TABLE IF EXISTS `category`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `category` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `version` bigint(20) NOT NULL,
  `category_name` varchar(255) CHARACTER SET utf8mb3 COLLATE utf8mb3_unicode_ci NOT NULL,
  `is_disabled` bit(1) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `category_name` (`category_name`)
) ENGINE=MyISAM AUTO_INCREMENT=38 DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `category_link`
--

DROP TABLE IF EXISTS `category_link`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `category_link` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `version` bigint(20) NOT NULL,
  `category_id` bigint(20) NOT NULL,
  `synset_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `FK142F1A9BA3843755` (`synset_id`),
  KEY `FK142F1A9B6197A215` (`category_id`)
) ENGINE=MyISAM AUTO_INCREMENT=27025 DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `contracts`
--

DROP TABLE IF EXISTS `contracts`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `contracts` (
  `address` varchar(42) NOT NULL,
  `bytecode` longtext DEFAULT NULL,
  `function_sighashes` longtext DEFAULT NULL,
  `is_erc20` tinyint(1) DEFAULT NULL,
  `is_erc721` tinyint(1) DEFAULT NULL,
  `block_number` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`address`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cov_hosp`
--

DROP TABLE IF EXISTS `cov_hosp`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cov_hosp` (
  `untry` varchar(11) DEFAULT NULL,
  `indicator` varchar(39) DEFAULT NULL,
  `date` varchar(10) DEFAULT NULL,
  `year_week` varchar(8) DEFAULT NULL,
  `value` varchar(19) DEFAULT NULL,
  `source` varchar(36) DEFAULT NULL,
  `url` varchar(157) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cov_owid2`
--

DROP TABLE IF EXISTS `cov_owid2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cov_owid2` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `iso_code` varchar(33) DEFAULT NULL,
  `continent` varchar(33) DEFAULT NULL,
  `location` varchar(33) DEFAULT NULL,
  `date` varchar(33) DEFAULT NULL,
  `total_cases` double DEFAULT NULL,
  `new_cases` double DEFAULT NULL,
  `new_cases_smoothed` double DEFAULT NULL,
  `total_deaths` double DEFAULT NULL,
  `new_deaths` double DEFAULT NULL,
  `new_deaths_smoothed` double DEFAULT NULL,
  `total_cases_per_million` double DEFAULT NULL,
  `new_cases_per_million` double DEFAULT NULL,
  `new_cases_smoothed_per_million` double DEFAULT NULL,
  `total_deaths_per_million` double DEFAULT NULL,
  `new_deaths_per_million` double DEFAULT NULL,
  `new_deaths_smoothed_per_million` double DEFAULT NULL,
  `reproduction_rate` double DEFAULT NULL,
  `icu_patients` double DEFAULT NULL,
  `icu_patients_per_million` double DEFAULT NULL,
  `hosp_patients` double DEFAULT NULL,
  `hosp_patients_per_million` double DEFAULT NULL,
  `weekly_icu_admissions` double DEFAULT NULL,
  `weekly_icu_admissions_per_million` double DEFAULT NULL,
  `weekly_hosp_admissions` double DEFAULT NULL,
  `weekly_hosp_admissions_per_million` double DEFAULT NULL,
  `new_tests` double DEFAULT NULL,
  `total_tests` double DEFAULT NULL,
  `total_tests_per_thousand` double DEFAULT NULL,
  `new_tests_per_thousand` double DEFAULT NULL,
  `new_tests_smoothed` double DEFAULT NULL,
  `new_tests_smoothed_per_thousand` double DEFAULT NULL,
  `positive_rate` double DEFAULT NULL,
  `tests_per_case` double DEFAULT NULL,
  `tests_units` text DEFAULT NULL,
  `total_vaccinations` double DEFAULT NULL,
  `people_vaccinated` double DEFAULT NULL,
  `people_fully_vaccinated` double DEFAULT NULL,
  `new_vaccinations` double DEFAULT NULL,
  `new_vaccinations_smoothed` double DEFAULT NULL,
  `total_vaccinations_per_hundred` double DEFAULT NULL,
  `people_vaccinated_per_hundred` double DEFAULT NULL,
  `people_fully_vaccinated_per_hundred` double DEFAULT NULL,
  `new_vaccinations_smoothed_per_million` double DEFAULT NULL,
  `stringency_index` double DEFAULT NULL,
  `population` double DEFAULT NULL,
  `population_density` double DEFAULT NULL,
  `median_age` double DEFAULT NULL,
  `aged_65_older` double DEFAULT NULL,
  `aged_70_older` double DEFAULT NULL,
  `gdp_per_capita` double DEFAULT NULL,
  `extreme_poverty` double DEFAULT NULL,
  `cardiovasc_death_rate` double DEFAULT NULL,
  `diabetes_prevalence` double DEFAULT NULL,
  `female_smokers` double DEFAULT NULL,
  `male_smokers` double DEFAULT NULL,
  `handwashing_facilities` double DEFAULT NULL,
  `hospital_beds_per_thousand` double DEFAULT NULL,
  `life_expectancy` double DEFAULT NULL,
  `human_development_index` double DEFAULT NULL,
  `excess_mortality` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cov_vacc`
--

DROP TABLE IF EXISTS `cov_vacc`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cov_vacc` (
  `YearWeekISO` varchar(8) DEFAULT NULL,
  `FirstDose` int(7) DEFAULT NULL,
  `FirstDoseRefused` varchar(4) DEFAULT NULL,
  `SecondDose` int(6) DEFAULT NULL,
  `UnknownDose` int(6) DEFAULT NULL,
  `NumberDosesReceived` varchar(7) DEFAULT NULL,
  `Region` varchar(7) DEFAULT NULL,
  `Population` int(8) DEFAULT NULL,
  `ReportingCountry` varchar(2) DEFAULT NULL,
  `TargetGroup` varchar(8) DEFAULT NULL,
  `Vaccine` varchar(6) DEFAULT NULL,
  `Denominator` varchar(8) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cov_variant`
--

DROP TABLE IF EXISTS `cov_variant`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cov_variant` (
  `untry` varchar(13) DEFAULT NULL,
  `country_code` varchar(2) DEFAULT NULL,
  `year_week` varchar(7) DEFAULT NULL,
  `source` varchar(6) DEFAULT NULL,
  `new_cases` int(6) DEFAULT NULL,
  `number_sequenced` varchar(4) DEFAULT NULL,
  `percent_cases_sequenced` varchar(4) DEFAULT NULL,
  `valid_denominator` varchar(3) DEFAULT NULL,
  `variant` varchar(15) DEFAULT NULL,
  `number_detections_variant` int(4) DEFAULT NULL,
  `percent_variant` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `easun_data`
--

DROP TABLE IF EXISTS `easun_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `easun_data` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `timestamp` datetime DEFAULT current_timestamp(),
  `grid_voltage` decimal(5,1) DEFAULT NULL,
  `battery_voltage` decimal(5,1) DEFAULT NULL,
  `battery_discharge_current` int(11) DEFAULT NULL,
  `battery_capacity` int(11) DEFAULT NULL,
  `ac_output_active_power` int(11) DEFAULT NULL,
  `output_load_percent` int(11) DEFAULT NULL,
  `line_power_direction` varchar(20) DEFAULT NULL,
  `working_mode` varchar(20) DEFAULT NULL,
  `battery_type` varchar(20) DEFAULT NULL,
  `battery_bulk_voltage` decimal(5,1) DEFAULT NULL,
  `battery_float_voltage` decimal(5,1) DEFAULT NULL,
  `max_charging_current` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `epex_prices`
--

DROP TABLE IF EXISTS `epex_prices`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `epex_prices` (
  `delivery_date` date NOT NULL,
  `hour` int(11) NOT NULL,
  `price_eur_mwh` float DEFAULT NULL,
  PRIMARY KEY (`delivery_date`,`hour`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `error_values`
--

DROP TABLE IF EXISTS `error_values`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `error_values` (
  `operating_id` bigint(20) DEFAULT NULL,
  `error_code` text DEFAULT NULL,
  `is_active` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `excesslog`
--

DROP TABLE IF EXISTS `excesslog`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `excesslog` (
  `timestamp` datetime NOT NULL,
  `scriptname` varchar(50) NOT NULL,
  `decision` int(11) NOT NULL,
  `decisions` varchar(255) DEFAULT NULL,
  `pccwatt` float NOT NULL,
  `excesswatt` float NOT NULL,
  `request` int(11) DEFAULT NULL,
  PRIMARY KEY (`timestamp`,`scriptname`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `garage`
--

DROP TABLE IF EXISTS `garage`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `garage` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `date_insert` datetime DEFAULT NULL,
  `date_open` varchar(33) DEFAULT NULL,
  `date_close` varchar(33) DEFAULT NULL,
  `time_open` varchar(33) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1035 DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `heatpr290`
--

DROP TABLE IF EXISTS `heatpr290`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `heatpr290` (
  `hot_water_active` tinyint(1) DEFAULT NULL,
  `reserved` text DEFAULT NULL,
  `heating_active` tinyint(1) DEFAULT NULL,
  `cooling_active` tinyint(1) DEFAULT NULL,
  `dc_fan_1_valid` tinyint(1) DEFAULT NULL,
  `dc_fan_2_valid` tinyint(1) DEFAULT NULL,
  `defrosting_active` tinyint(1) DEFAULT NULL,
  `compressor` tinyint(1) DEFAULT NULL,
  `fan_motor` tinyint(1) DEFAULT NULL,
  `four-way_valve` tinyint(1) DEFAULT NULL,
  `chassis_electric_heating` tinyint(1) DEFAULT NULL,
  `a/c_electric_heating` tinyint(1) DEFAULT NULL,
  `three-way_valve` tinyint(1) DEFAULT NULL,
  `water_tank_electric_heating` tinyint(1) DEFAULT NULL,
  `circulation_pump` tinyint(1) DEFAULT NULL,
  `crankshaft_electric_heating` tinyint(1) DEFAULT NULL,
  `fault_flag_1` text DEFAULT NULL,
  `fault_flag_2` text DEFAULT NULL,
  `fault_flag_3` text DEFAULT NULL,
  `fault_flag_4` text DEFAULT NULL,
  `fault_flag_5` text DEFAULT NULL,
  `fault_flag_6` text DEFAULT NULL,
  `fault_flag_7` text DEFAULT NULL,
  `inlet_water_temperature` text DEFAULT NULL,
  `water_tank_temperature` text DEFAULT NULL,
  `ambient_temperature` text DEFAULT NULL,
  `outlet_water_temperature` text DEFAULT NULL,
  `suction_gas_temperature` text DEFAULT NULL,
  `external_coil_temperature` text DEFAULT NULL,
  `inner_coil_temperature` text DEFAULT NULL,
  `exhaust_gas_temperature` text DEFAULT NULL,
  `compressor_actual_frequency` text DEFAULT NULL,
  `compressor_current` text DEFAULT NULL,
  `low_pressure_conversion_temperature` text DEFAULT NULL,
  `dc_water_pump_speed` text DEFAULT NULL,
  `low_pressure_value` text DEFAULT NULL,
  `compressor_operating_power` text DEFAULT NULL,
  `parameter_flag_1` text DEFAULT NULL,
  `control_flag_1` text DEFAULT NULL,
  `control_flag_2` text DEFAULT NULL,
  `mode` text DEFAULT NULL,
  `defrost_frequency` text DEFAULT NULL,
  `defrost_cycle` text DEFAULT NULL,
  `defrost_time` text DEFAULT NULL,
  `action_cycle_of_main_expansion_valve` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `heatpump`
--

DROP TABLE IF EXISTS `heatpump`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `heatpump` (
  `id` int(13) NOT NULL,
  `code` varchar(19) DEFAULT NULL,
  `val` int(11) DEFAULT NULL,
  `device` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `heatpump2`
--

DROP TABLE IF EXISTS `heatpump2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `heatpump2` (
  `id` mediumint(9) NOT NULL AUTO_INCREMENT,
  `dt` datetime DEFAULT NULL,
  `switch` int(11) DEFAULT NULL,
  `mode` varchar(9) DEFAULT NULL,
  `work_mode` varchar(9) DEFAULT NULL,
  `temp_unit_convert` varchar(9) DEFAULT NULL,
  `fault` int(11) DEFAULT NULL,
  `intemp` int(11) DEFAULT NULL,
  `outtemp` int(11) DEFAULT NULL,
  `whjtemp` int(11) DEFAULT NULL,
  `cmptemp` int(11) DEFAULT NULL,
  `hqtemp` int(11) DEFAULT NULL,
  `ptemp` int(11) DEFAULT NULL,
  `cptemp` int(11) DEFAULT NULL,
  `wttemp` int(11) DEFAULT NULL,
  `step_run` int(11) DEFAULT NULL,
  `stepb_run` int(11) DEFAULT NULL,
  `cmp_cur` int(11) DEFAULT NULL,
  `sink_temp` int(11) DEFAULT NULL,
  `dc_bus_voltage` int(11) DEFAULT NULL,
  `cmp_act_frep` int(11) DEFAULT NULL,
  `dc_fan_speed` int(11) DEFAULT NULL,
  `dc_fan2_speed` int(11) DEFAULT NULL,
  `voltage` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=306606 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `heatpump3`
--

DROP TABLE IF EXISTS `heatpump3`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `heatpump3` (
  `id` mediumint(9) NOT NULL AUTO_INCREMENT,
  `dt` datetime DEFAULT NULL,
  `v1` int(11) DEFAULT NULL,
  `v2` varchar(9) DEFAULT NULL,
  `v5` varchar(9) DEFAULT NULL,
  `v6` varchar(9) DEFAULT NULL,
  `v15` int(11) DEFAULT NULL,
  `v101` int(11) DEFAULT NULL,
  `v102` int(11) DEFAULT NULL,
  `v103` int(11) DEFAULT NULL,
  `v104` int(11) DEFAULT NULL,
  `v015` int(11) DEFAULT NULL,
  `v106` int(11) DEFAULT NULL,
  `v107` int(11) DEFAULT NULL,
  `v108` int(11) DEFAULT NULL,
  `v109` int(11) DEFAULT NULL,
  `v111` int(11) DEFAULT NULL,
  `v112` int(11) DEFAULT NULL,
  `v113` int(11) DEFAULT NULL,
  `v115` int(11) DEFAULT NULL,
  `v116` int(11) DEFAULT NULL,
  `v117` int(11) DEFAULT NULL,
  `v3` varchar(9) DEFAULT NULL,
  `v4` varchar(9) DEFAULT NULL,
  `v105` int(11) DEFAULT NULL,
  `v114` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=296423 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `heizung`
--

DROP TABLE IF EXISTS `heizung`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `heizung` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `F1` float DEFAULT NULL,
  `S1` varchar(8) CHARACTER SET latin1 COLLATE latin1_swedish_ci DEFAULT NULL,
  `DT` datetime DEFAULT NULL,
  `I1` int(11) DEFAULT NULL,
  `I2` int(11) DEFAULT NULL,
  `I3` int(11) DEFAULT NULL,
  `I4` int(11) DEFAULT NULL,
  `I5` int(11) DEFAULT NULL,
  `I6` int(11) DEFAULT NULL,
  `i7` int(11) DEFAULT NULL,
  `i8` int(11) DEFAULT NULL,
  `i9` int(11) DEFAULT NULL,
  `i10` int(11) DEFAULT NULL,
  `i11` int(11) DEFAULT NULL,
  `i12` int(11) DEFAULT NULL,
  `ky9awz` float DEFAULT NULL,
  `dth` datetime DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `heizung_DT_IDX` (`DT`),
  KEY `idx_heizung_dth_i6` (`dth`,`I6`)
) ENGINE=InnoDB AUTO_INCREMENT=2422993 DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `heizung2`
--

DROP TABLE IF EXISTS `heizung2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `heizung2` (
  `id` bigint(20) NOT NULL DEFAULT 0,
  `F1` float DEFAULT NULL,
  `S1` varchar(8) CHARACTER SET latin1 COLLATE latin1_swedish_ci DEFAULT NULL,
  `DT` datetime DEFAULT NULL,
  `I1` int(11) DEFAULT NULL,
  `I2` int(11) DEFAULT NULL,
  `I3` int(11) DEFAULT NULL,
  `I4` int(11) DEFAULT NULL,
  `I5` int(11) DEFAULT NULL,
  `I6` int(11) DEFAULT NULL,
  `i7` int(11) DEFAULT NULL,
  `i8` int(11) DEFAULT NULL,
  `i9` int(11) DEFAULT NULL,
  `i10` int(11) DEFAULT NULL,
  `i11` int(11) DEFAULT NULL,
  `i12` int(11) DEFAULT NULL,
  `ky9awz` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `heizungslog`
--

DROP TABLE IF EXISTS `heizungslog`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `heizungslog` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `zeit` datetime DEFAULT NULL,
  `fww` float DEFAULT NULL,
  `druck` float DEFAULT NULL,
  `solar` float DEFAULT NULL,
  `gruppe` varchar(20) DEFAULT NULL,
  `raw0` int(11) DEFAULT NULL,
  `raw1` int(11) DEFAULT NULL,
  `raw2` int(11) DEFAULT NULL,
  `raw3` int(11) DEFAULT NULL,
  `kwh` int(11) DEFAULT NULL,
  `brunnen_cnt` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `inputs`
--

DROP TABLE IF EXISTS `inputs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `inputs` (
  `transaction_hash` varchar(64) NOT NULL,
  `input_index` int(11) NOT NULL,
  `script_asm` longtext DEFAULT NULL,
  `script_hex` longtext DEFAULT NULL,
  `required_signatures` int(11) DEFAULT NULL,
  `type` varchar(100) DEFAULT NULL,
  `addresses` longtext DEFAULT NULL,
  `value` decimal(20,8) DEFAULT NULL,
  `block_month` int(6) NOT NULL,
  PRIMARY KEY (`transaction_hash`,`input_index`,`block_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
 PARTITION BY RANGE (`block_month`)
(PARTITION `p202512` VALUES LESS THAN (202601) ENGINE = InnoDB,
 PARTITION `p_future` VALUES LESS THAN MAXVALUE ENGINE = InnoDB);
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `inputs_test`
--

DROP TABLE IF EXISTS `inputs_test`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `inputs_test` (
  `transaction_hash` varchar(64) NOT NULL,
  `input_index` int(11) NOT NULL,
  `script_asm` longtext DEFAULT NULL,
  `script_hex` longtext DEFAULT NULL,
  `required_signatures` int(11) DEFAULT NULL,
  `type` varchar(100) DEFAULT NULL,
  `addresses` longtext DEFAULT NULL,
  `value` decimal(20,8) DEFAULT NULL,
  `block_month` int(6) NOT NULL,
  PRIMARY KEY (`transaction_hash`,`input_index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `inverter2`
--

DROP TABLE IF EXISTS `inverter2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `inverter2` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `LoadsPower` int(11) DEFAULT NULL,
  `loads` int(11) DEFAULT NULL,
  `generationPower` int(11) DEFAULT NULL,
  `generation` int(11) DEFAULT NULL,
  `feedinPower` int(11) DEFAULT NULL,
  `feedin` int(11) DEFAULT NULL,
  `feedin2` int(11) DEFAULT NULL,
  `input` int(11) DEFAULT NULL,
  `gridConsumptionPower` int(11) DEFAULT NULL,
  `gridConsumption` int(11) DEFAULT NULL,
  `gridConsumption2` int(11) DEFAULT NULL,
  `RVolt` int(11) DEFAULT NULL,
  `RCurrent` int(11) DEFAULT NULL,
  `RFreq` int(11) DEFAULT NULL,
  `RPower` int(11) DEFAULT NULL,
  `SVolt` int(11) DEFAULT NULL,
  `SCurrent` int(11) DEFAULT NULL,
  `SFreq` int(11) DEFAULT NULL,
  `SPower` int(11) DEFAULT NULL,
  `TVolt` int(11) DEFAULT NULL,
  `TCurrent` int(11) DEFAULT NULL,
  `TFreq` int(11) DEFAULT NULL,
  `TPower` int(11) DEFAULT NULL,
  `pvPower` int(11) DEFAULT NULL,
  `pv1Volt` int(11) DEFAULT NULL,
  `pv1Current` int(11) DEFAULT NULL,
  `pv1Power` int(11) DEFAULT NULL,
  `pv2Volt` int(11) DEFAULT NULL,
  `pv2Current` int(11) DEFAULT NULL,
  `pv2Power` int(11) DEFAULT NULL,
  `batcurrent` int(11) DEFAULT NULL,
  `ts` datetime DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `inverter2_ts_IDX` (`ts`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=1493425 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `inverter_data`
--

DROP TABLE IF EXISTS `inverter_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `inverter_data` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `device_id` varchar(16) DEFAULT NULL,
  `timestamp` datetime DEFAULT NULL,
  `section` varchar(255) DEFAULT NULL,
  `ActivePower_Load_L1N` float DEFAULT NULL,
  `ActivePower_Load_L2N` float DEFAULT NULL,
  `ActivePower_Load_R` float DEFAULT NULL,
  `ActivePower_Load_S` float DEFAULT NULL,
  `ActivePower_Load_Sys` float DEFAULT NULL,
  `ActivePower_Load_T` float DEFAULT NULL,
  `ActivePower_Load_Total` float DEFAULT NULL,
  `ActivePower_Output_Total` float DEFAULT NULL,
  `ActivePower_PCC_R` float DEFAULT NULL,
  `ActivePower_PCC_S` float DEFAULT NULL,
  `ActivePower_PCC_T` float DEFAULT NULL,
  `ActivePower_PCC_Total` float DEFAULT NULL,
  `ActivePower_PV_Ext` float DEFAULT NULL,
  `AddressMask_Realtime_ElectricityStatistics1` float DEFAULT NULL,
  `AddressMask_Realtime_EmergencyOutput1` float DEFAULT NULL,
  `AddressMask_Realtime_GridOutput1` float DEFAULT NULL,
  `AddressMask_Realtime_Input_Bat1` float DEFAULT NULL,
  `AddressMask_Realtime_Input_PV1` float DEFAULT NULL,
  `ApparentPower_Load_R` float DEFAULT NULL,
  `ApparentPower_Load_S` float DEFAULT NULL,
  `ApparentPower_Load_T` float DEFAULT NULL,
  `ApparentPower_Load_Total` float DEFAULT NULL,
  `Bat_Charge_Total` float DEFAULT NULL,
  `Bat_Discharge_Today` float DEFAULT NULL,
  `Bat_Discharge_Total` float DEFAULT NULL,
  `ChargeCycle_Bat1` float DEFAULT NULL,
  `Current_Bat1` float DEFAULT NULL,
  `Current_Load_L1N` float DEFAULT NULL,
  `Current_Load_L2N` float DEFAULT NULL,
  `Current_Load_R` float DEFAULT NULL,
  `Current_Load_S` float DEFAULT NULL,
  `Current_Load_T` float DEFAULT NULL,
  `Current_Output_R` float DEFAULT NULL,
  `Current_Output_S` float DEFAULT NULL,
  `Current_Output_T` float DEFAULT NULL,
  `Current_PCC_R` float DEFAULT NULL,
  `Current_PCC_S` float DEFAULT NULL,
  `Current_PCC_T` float DEFAULT NULL,
  `Current_PV1` float DEFAULT NULL,
  `Current_PV2` float DEFAULT NULL,
  `ESR_Rsvd2` float DEFAULT NULL,
  `ESS_Rsvd2` float DEFAULT NULL,
  `EST_Rsvd2` float DEFAULT NULL,
  `Energy_Purchase_Today` float DEFAULT NULL,
  `Energy_Purchase_Total` float DEFAULT NULL,
  `Energy_Selling_Today` float DEFAULT NULL,
  `Energy_Selling_Total` float DEFAULT NULL,
  `Frequency_Grid` float DEFAULT NULL,
  `Frequency_Output` float DEFAULT NULL,
  `LoadPeakRatio_R` float DEFAULT NULL,
  `LoadPeakRatio_S` float DEFAULT NULL,
  `LoadPeakRatio_T` float DEFAULT NULL,
  `Load_Consumption_Today` float DEFAULT NULL,
  `Load_Consumption_Total` float DEFAULT NULL,
  `PV_Generation_Today` float DEFAULT NULL,
  `PV_Generation_Total` float DEFAULT NULL,
  `Power_Bat1` float DEFAULT NULL,
  `Power_PV1` float DEFAULT NULL,
  `Power_PV2` float DEFAULT NULL,
  `ReactivePower_Load_R` float DEFAULT NULL,
  `ReactivePower_Load_S` float DEFAULT NULL,
  `ReactivePower_Load_T` float DEFAULT NULL,
  `ReactivePower_Load_Total` float DEFAULT NULL,
  `SOC_Bat1` float DEFAULT NULL,
  `SOH_Bat1` float DEFAULT NULL,
  `Temperature_Env_Bat1` float DEFAULT NULL,
  `Voltage_Bat1` float DEFAULT NULL,
  `Voltage_Load_R` float DEFAULT NULL,
  `Voltage_Load_S` float DEFAULT NULL,
  `Voltage_Load_T` float DEFAULT NULL,
  `Voltage_Output_L1N` float DEFAULT NULL,
  `Voltage_Output_L2N` float DEFAULT NULL,
  `Voltage_Output_R` float DEFAULT NULL,
  `Voltage_Output_S` float DEFAULT NULL,
  `Voltage_Output_T` float DEFAULT NULL,
  `Voltage_PV1` float DEFAULT NULL,
  `Voltage_PV2` float DEFAULT NULL,
  `Voltage_Phase_R` float DEFAULT NULL,
  `Voltage_Phase_S` float DEFAULT NULL,
  `Voltage_Phase_T` float DEFAULT NULL,
  `Bat_Charge_Today` float DEFAULT NULL,
  `ESOutput_Rsvd1` float DEFAULT NULL,
  `PV_Generation_Year` float DEFAULT NULL,
  `Energy_Purchase_Month` float DEFAULT NULL,
  `Energy_Selling_Month` float DEFAULT NULL,
  `Bat_Charge_Year` float DEFAULT NULL,
  `R_Rsvd1` float DEFAULT NULL,
  `ReactivePower_PCC_Total` float DEFAULT NULL,
  `ReactivePower_Output_Total` float DEFAULT NULL,
  `ApparentPower_PCC_Total` float DEFAULT NULL,
  `ApparentPower_Output_Total` float DEFAULT NULL,
  `GridOutput_Rsvd1` float DEFAULT NULL,
  `ActivePower_Output_S` float DEFAULT NULL,
  `ActivePower_Output_T` float DEFAULT NULL,
  `PowerFactor_Output_T` float DEFAULT NULL,
  `S_Rsvd1` float DEFAULT NULL,
  `ActivePower_Output_R` float DEFAULT NULL,
  `ReactivePower_Output_T` float DEFAULT NULL,
  `PowerFactor_Output_S` float DEFAULT NULL,
  `ReactivePower_PCC_S` float DEFAULT NULL,
  `ReactivePower_Output_R` float DEFAULT NULL,
  `PowerFactor_PCC_R` float DEFAULT NULL,
  `ReactivePower_PCC_R` float DEFAULT NULL,
  `ReactivePower_Output_S` float DEFAULT NULL,
  `PowerFactor_PCC_S` float DEFAULT NULL,
  `PowerFactor_Output_R` float DEFAULT NULL,
  `AddressMask_General1` float DEFAULT NULL,
  `Modbus_Protocol_Version` float DEFAULT NULL,
  `SysTime_Second` float DEFAULT NULL,
  `RS485Config_Address` float DEFAULT NULL,
  `TOU_ID` float DEFAULT NULL,
  `Parallel_Control` float DEFAULT NULL,
  `Reactive_Power_Response_Time` float DEFAULT NULL,
  `Temperature_Env1` float DEFAULT NULL,
  `Timing_Power_Discharge` float DEFAULT NULL,
  `GenerationTime_Total` float DEFAULT NULL,
  `Italay_Autotest_Result32` float DEFAULT NULL,
  `AntiReflux_Control` float DEFAULT NULL,
  `EnergyStatistics_Config` float DEFAULT NULL,
  `Italay_Autotest_Result16` float DEFAULT NULL,
  `TOU_Charge_Start` float DEFAULT NULL,
  `Active_Power_Limit_Speed` float DEFAULT NULL,
  `Italay_Autotest_Result21` float DEFAULT NULL,
  `Timing_Control` float DEFAULT NULL,
  `TOU_Executed_Date_Start` float DEFAULT NULL,
  `Italay_Autotest_Result20` float DEFAULT NULL,
  `TOU_Executed_Day_of_Week` float DEFAULT NULL,
  `IVCurveScan_Control` float DEFAULT NULL,
  `GridVoltageDropMinPower` float DEFAULT NULL,
  `Power_Factor_Setting` float DEFAULT NULL,
  `ChgDerateVoltEnd` float DEFAULT NULL,
  `SysTime_Month` float DEFAULT NULL,
  `Energy_Storage_Mode_Control` float DEFAULT NULL,
  `Reactive_Power_Setting` float DEFAULT NULL,
  `UnbalancedSupport_Control` float DEFAULT NULL,
  `InputType_Control` float DEFAULT NULL,
  `Italay_Autotest_Result14` float DEFAULT NULL,
  `SysTimeConfig_Control` float DEFAULT NULL,
  `Italay_Autotest_Result10` float DEFAULT NULL,
  `Italay_Autotest_Result8` float DEFAULT NULL,
  `TOU_Charge_Target_SOC` float DEFAULT NULL,
  `VoltageDerateResponseTime` float DEFAULT NULL,
  `Italay_Autotest_Result26` float DEFAULT NULL,
  `Italay_Autotest_Result6` float DEFAULT NULL,
  `InsulationResistance` float DEFAULT NULL,
  `TOU_Executed_Date_End` float DEFAULT NULL,
  `Italay_Autotest_Result13` float DEFAULT NULL,
  `SysTimeConfig_Second` float DEFAULT NULL,
  `Timing_Power_Charge` float DEFAULT NULL,
  `OvervoltageDownSpeed` float DEFAULT NULL,
  `TOU_Control` float DEFAULT NULL,
  `SysTime_Hour` float DEFAULT NULL,
  `EnergyStatistics_Date_Year` float DEFAULT NULL,
  `Italay_Autotest_Result17` float DEFAULT NULL,
  `Italay_Autotest_Result24` float DEFAULT NULL,
  `Italay_Autotest_Result1` float DEFAULT NULL,
  `Timing_Charge_End` float DEFAULT NULL,
  `Italay_Autotest_Result11` float DEFAULT NULL,
  `Italay_Autotest_Result9` float DEFAULT NULL,
  `Peak_Shaving_Charge_Threshold` float DEFAULT NULL,
  `SysState` float DEFAULT NULL,
  `ChgDerateVoltStart` float DEFAULT NULL,
  `Italay_Autotest_Result7` float DEFAULT NULL,
  `Language` float DEFAULT NULL,
  `RS485Config_ParityBit` float DEFAULT NULL,
  `ServiceTime_Total` float DEFAULT NULL,
  `TOU_Charge_Power` float DEFAULT NULL,
  `SysTimeConfig_Month` float DEFAULT NULL,
  `Italay_Autotest_Result22` float DEFAULT NULL,
  `Italay_Autotest_Result12` float DEFAULT NULL,
  `ActiveOutputLimit` float DEFAULT NULL,
  `LogicReloadSpeed` float DEFAULT NULL,
  `RefluxPower` float DEFAULT NULL,
  `Italay_Autotest_Result27` float DEFAULT NULL,
  `Italay_Autotest_Result29` float DEFAULT NULL,
  `Active_Power_Export_Limit` float DEFAULT NULL,
  `Italay_Autotest_Result33` float DEFAULT NULL,
  `Timing_Charge_Start` float DEFAULT NULL,
  `Peak_Shaving_Discharge_Threshold` float DEFAULT NULL,
  `SysTime_Minute` float DEFAULT NULL,
  `AddressMask_Config_Basic1` float DEFAULT NULL,
  `Countdown` float DEFAULT NULL,
  `Italay_Autotest_Result5` float DEFAULT NULL,
  `Italay_Autotest_Result19` float DEFAULT NULL,
  `SysTimeConfig_Date` float DEFAULT NULL,
  `Italay_Autotest_Result23` float DEFAULT NULL,
  `TOU_Rsvd1` float DEFAULT NULL,
  `Italay_Autotest_Result18` float DEFAULT NULL,
  `Fault19` float DEFAULT NULL,
  `LogicInterface_Control` float DEFAULT NULL,
  `Power_Control` float DEFAULT NULL,
  `SysTime_Date` float DEFAULT NULL,
  `TOU_On_Off_Control` float DEFAULT NULL,
  `AddressMask_Config_ReadOnly_Result1` float DEFAULT NULL,
  `Italay_Autotest_Result15` float DEFAULT NULL,
  `Active_Power_Import_Limit` float DEFAULT NULL,
  `Parallel_Master_Slave` float DEFAULT NULL,
  `SysTimeConfig_Year` float DEFAULT NULL,
  `Temperature_HeatSink1` float DEFAULT NULL,
  `AddressMask_Realtime_CombinerInfo1` float DEFAULT NULL,
  `EnergyStatistics_Date_Month` float DEFAULT NULL,
  `ReconnectPowerUpSpeed` float DEFAULT NULL,
  `EnergyStatistics_Date_Date` float DEFAULT NULL,
  `InputType_Channel0_Config` float DEFAULT NULL,
  `Italay_Autotest_Result3` float DEFAULT NULL,
  `RS485Config_Control` float DEFAULT NULL,
  `IVCurveScan_Period` float DEFAULT NULL,
  `IVCurveScan_ReadChannel` float DEFAULT NULL,
  `SysTimeConfig_Minute` float DEFAULT NULL,
  `Timing_ID` float DEFAULT NULL,
  `Timing_Discharge_End` float DEFAULT NULL,
  `Timing_Rsvd1` float DEFAULT NULL,
  `SysTimeConfig_Hour` float DEFAULT NULL,
  `SysTime_Year` float DEFAULT NULL,
  `Italay_Autotest_Result25` float DEFAULT NULL,
  `Parallel_Address` float DEFAULT NULL,
  `Italay_Autotest_Result31` float DEFAULT NULL,
  `Italy_AutoTest` float DEFAULT NULL,
  `CT_Auto_Calibrate` float DEFAULT NULL,
  `Temperature_Inv1` float DEFAULT NULL,
  `AddressMask_Realtime_SysInfo1` float DEFAULT NULL,
  `Italay_Autotest_Result30` float DEFAULT NULL,
  `GenerationTime_Today` float DEFAULT NULL,
  `RS485Config_StopBit` float DEFAULT NULL,
  `EPS_Control` float DEFAULT NULL,
  `Italay_Autotest_Result2` float DEFAULT NULL,
  `AntiReflux_Power` float DEFAULT NULL,
  `IVCurveScan_Oneshot` float DEFAULT NULL,
  `AddressMask_Config_Remote1` float DEFAULT NULL,
  `TOU_Charge_End` float DEFAULT NULL,
  `RS485Config_Baud` float DEFAULT NULL,
  `Timing_Discharge_Start` float DEFAULT NULL,
  `Fault1` float DEFAULT NULL,
  `Timing_On_Off_Control` float DEFAULT NULL,
  `Remote_On_Off_Control` float DEFAULT NULL,
  `SafetyUpdateFromUSB_Control` float DEFAULT NULL,
  `Italay_Autotest_Result28` float DEFAULT NULL,
  `Italay_Autotest_Result4` float DEFAULT NULL,
  `SVG_Fixed_Reactive_Power_Setting` float DEFAULT NULL,
  `Local_Upgrade_Control` float DEFAULT NULL,
  `Remote_Upgrade_Control` float DEFAULT NULL,
  `Local_Upgrade_Status` float DEFAULT NULL,
  `AddressMask_Config_Core1` float DEFAULT NULL,
  `Factory_Reset` float DEFAULT NULL,
  `Country_Code` float DEFAULT NULL,
  `Safety_Version` float DEFAULT NULL,
  `PCU_Hardware_Version` float DEFAULT NULL,
  `PCU_DC_Low_Voltage` float DEFAULT NULL,
  `Sofar_BMS_Version_0x9019` float DEFAULT NULL,
  `Total_Current` float DEFAULT NULL,
  `PCU_Low_Power` float DEFAULT NULL,
  `BMS_Sys_Alarm0` float DEFAULT NULL,
  `BMS_Sys_Protect0` float DEFAULT NULL,
  `PCU_Low_Current` float DEFAULT NULL,
  `BMS_Sys_Time` float DEFAULT NULL,
  `BaPack_Number` float DEFAULT NULL,
  `PCU_Can1_Version` float DEFAULT NULL,
  `SOC` float DEFAULT NULL,
  `AddressMask_PCU1` float DEFAULT NULL,
  `Sofar_BMS_Version_0x901A` float DEFAULT NULL,
  `PCU_Can2_Version` float DEFAULT NULL,
  `Sofar_BMS_Version_0x9018` float DEFAULT NULL,
  `PCU_Radiator_Temperature2` float DEFAULT NULL,
  `BMS_Version` float DEFAULT NULL,
  `PCU_Warning1` float DEFAULT NULL,
  `PCU_Internal_Temperature` float DEFAULT NULL,
  `SOH` float DEFAULT NULL,
  `AddressMask_BMS1_System` float DEFAULT NULL,
  `BMS_Manufacture_Name1` float DEFAULT NULL,
  `Cell_Average_Temperature` float DEFAULT NULL,
  `PCU_Fault1` float DEFAULT NULL,
  `BMS_CAN_Version` float DEFAULT NULL,
  `PCU_DC_High_Voltage` float DEFAULT NULL,
  `Realtime_Capacity` float DEFAULT NULL,
  `PCU_ID` float DEFAULT NULL,
  `Cell_Type` float DEFAULT NULL,
  `BMS_Inquire` float DEFAULT NULL,
  `PCU_Software_Version1` float DEFAULT NULL,
  `PCU_Radiator_Temperature1` float DEFAULT NULL,
  `Total_Voltage` float DEFAULT NULL,
  `BMS_Manufacture_Name0` float DEFAULT NULL,
  `Sofar_BMS_Version_0x901B` float DEFAULT NULL,
  `PCU_Number` float DEFAULT NULL,
  `PCU_Work_State` float DEFAULT NULL,
  `LogicDerateSpeed` float DEFAULT NULL,
  `RefluxOVloadTime` float DEFAULT NULL,
  `Voltage_Line_L2` float DEFAULT NULL,
  `ActivePower_Output_L2N` float DEFAULT NULL,
  `Voltage_Line_L3` float DEFAULT NULL,
  `ReactivePower_PCC_T` float DEFAULT NULL,
  `Voltage_Phase_L1N` float DEFAULT NULL,
  `ActivePower_Output_L1N` float DEFAULT NULL,
  `Current_Output_L2N` float DEFAULT NULL,
  `ActivePower_PCC_L2N` float DEFAULT NULL,
  `Power_Factor` float DEFAULT NULL,
  `Current_PCC_L2N` float DEFAULT NULL,
  `PowerFactor_PCC_T` float DEFAULT NULL,
  `Voltage_Phase_L2N` float DEFAULT NULL,
  `T_Rsvd1` float DEFAULT NULL,
  `Current_PCC_L1N` float DEFAULT NULL,
  `ActivePower_PCC_L1N` float DEFAULT NULL,
  `Voltage_Line_L1` float DEFAULT NULL,
  `Current_Output_L1N` float DEFAULT NULL,
  `socmaxtime` time DEFAULT NULL,
  `ts_min` datetime GENERATED ALWAYS AS ('2024-01-01' + interval timestampdiff(MINUTE,'2024-01-01',`timestamp`) minute) VIRTUAL,
  PRIMARY KEY (`id`),
  KEY `idx_inverter_data_timestamp` (`timestamp`),
  KEY `idx_inverter_data_timestamp_selling` (`timestamp`,`Energy_Selling_Today`),
  KEY `idx_inverter_data_timestamp_purchase` (`timestamp`,`Energy_Purchase_Today`),
  KEY `idx_timestamp_activepower` (`timestamp`,`ActivePower_PCC_Total`),
  KEY `ts_min` (`ts_min`)
) ENGINE=InnoDB AUTO_INCREMENT=802884 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `language`
--

DROP TABLE IF EXISTS `language`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `language` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `version` bigint(20) NOT NULL,
  `long_form` varchar(255) NOT NULL,
  `short_form` varchar(255) NOT NULL,
  `is_disabled` bit(1) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `long_form` (`long_form`),
  UNIQUE KEY `short_form` (`short_form`)
) ENGINE=MyISAM AUTO_INCREMENT=3 DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `link_type`
--

DROP TABLE IF EXISTS `link_type`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `link_type` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `version` bigint(20) NOT NULL,
  `link_name` varchar(255) NOT NULL,
  `other_direction_link_name` varchar(255) NOT NULL,
  `verb_name` varchar(255) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `link_name` (`link_name`),
  UNIQUE KEY `verb_name` (`verb_name`)
) ENGINE=MyISAM AUTO_INCREMENT=3 DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `macon_pivot`
--

DROP TABLE IF EXISTS `macon_pivot`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `macon_pivot` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `timestamp` datetime NOT NULL,
  `volumeflow` float DEFAULT NULL COMMENT 'Volumenstrom [L/h]',
  `unit_on_off` int(11) DEFAULT NULL COMMENT 'Register 2000 []',
  `set_frequency` int(11) DEFAULT NULL COMMENT 'Register 2057 [Hz]',
  `real_frequency` int(11) DEFAULT NULL COMMENT 'Register 2118 [Hz]',
  `ac_current` int(11) DEFAULT NULL COMMENT 'Register 2121 [A]',
  `system_status_2` int(11) DEFAULT NULL COMMENT 'Register 2135 [bits]',
  PRIMARY KEY (`id`),
  KEY `idx_timestamp` (`timestamp`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `macon_registers`
--

DROP TABLE IF EXISTS `macon_registers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `macon_registers` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `timestamp` datetime NOT NULL,
  `register` int(11) NOT NULL,
  `name` varchar(100) DEFAULT NULL,
  `value` float DEFAULT NULL,
  `unit` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3786 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mbus`
--

DROP TABLE IF EXISTS `mbus`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `mbus` (
  `id` mediumint(9) NOT NULL AUTO_INCREMENT,
  `Fabricationnumber` varchar(33) DEFAULT NULL,
  `Energy` int(11) DEFAULT NULL,
  `TimePoint1` varchar(11) DEFAULT NULL,
  `TimePoint2` varchar(11) DEFAULT NULL,
  `Energy2` int(11) DEFAULT NULL,
  `TimePoint3` varchar(11) DEFAULT NULL,
  `Energy3` int(11) DEFAULT NULL,
  `Volume` int(11) DEFAULT NULL,
  `Flowtemperature` int(11) DEFAULT NULL,
  `Returntemperature` int(11) DEFAULT NULL,
  `TemperatureDifference` int(11) DEFAULT NULL,
  `Power100W` int(11) DEFAULT NULL,
  `Volumeflow` int(11) DEFAULT NULL,
  `TimeStamp` varchar(22) DEFAULT NULL,
  `Operatingh` int(11) DEFAULT NULL,
  `Errorflags` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=19 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mbus2`
--

DROP TABLE IF EXISTS `mbus2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `mbus2` (
  `id` mediumint(9) NOT NULL AUTO_INCREMENT,
  `dt` datetime DEFAULT NULL,
  `Fabricationnumber` varchar(33) DEFAULT NULL,
  `Energy` int(11) DEFAULT NULL,
  `TimePoint1` varchar(11) DEFAULT NULL,
  `TimePoint2` varchar(11) DEFAULT NULL,
  `Energy2` int(11) DEFAULT NULL,
  `TimePoint3` varchar(11) DEFAULT NULL,
  `Energy3` int(11) DEFAULT NULL,
  `Volume` int(11) DEFAULT NULL,
  `Flowtemperature` int(11) DEFAULT NULL,
  `Returntemperature` int(11) DEFAULT NULL,
  `TemperatureDifference` int(11) DEFAULT NULL,
  `Power100W` int(11) DEFAULT NULL,
  `Volumeflow` int(11) DEFAULT NULL,
  `TimeStamp` varchar(22) DEFAULT NULL,
  `Operatingh` int(11) DEFAULT NULL,
  `Errorflags` int(11) DEFAULT NULL,
  `Energy1` varchar(11) DEFAULT NULL,
  `dth` varchar(13) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `mbus2_dt_IDX` (`dt`) USING BTREE,
  KEY `i_dth` (`dth`),
  KEY `idx_mbus2_dth` (`dth`)
) ENGINE=InnoDB AUTO_INCREMENT=259543 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `modbus_data`
--

DROP TABLE IF EXISTS `modbus_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `modbus_data` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `Name` varchar(255) DEFAULT NULL,
  `Value` text DEFAULT NULL,
  `Unit` varchar(50) DEFAULT NULL,
  `Type` varchar(50) DEFAULT NULL,
  `dt` datetime DEFAULT NULL,
  `dth` varchar(13) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2158333 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `outputs`
--

DROP TABLE IF EXISTS `outputs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `outputs` (
  `transaction_hash` varchar(64) NOT NULL,
  `output_index` int(11) NOT NULL,
  `script_asm` longtext DEFAULT NULL,
  `script_hex` longtext DEFAULT NULL,
  `required_signatures` int(11) DEFAULT NULL,
  `type` varchar(100) DEFAULT NULL,
  `addresses` longtext DEFAULT NULL,
  `value` decimal(20,8) DEFAULT NULL,
  `block_month` int(6) NOT NULL,
  PRIMARY KEY (`transaction_hash`,`output_index`,`block_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
 PARTITION BY RANGE (`block_month`)
(PARTITION `p202512` VALUES LESS THAN (202601) ENGINE = InnoDB,
 PARTITION `p_future` VALUES LESS THAN MAXVALUE ENGINE = InnoDB);
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `outputs_test`
--

DROP TABLE IF EXISTS `outputs_test`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `outputs_test` (
  `transaction_hash` varchar(64) NOT NULL,
  `output_index` int(11) NOT NULL,
  `script_asm` longtext DEFAULT NULL,
  `script_hex` longtext DEFAULT NULL,
  `required_signatures` int(11) DEFAULT NULL,
  `type` varchar(100) DEFAULT NULL,
  `addresses` longtext DEFAULT NULL,
  `value` decimal(20,8) DEFAULT NULL,
  `block_month` int(6) NOT NULL,
  PRIMARY KEY (`transaction_hash`,`output_index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pivot_table`
--

DROP TABLE IF EXISTS `pivot_table`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `pivot_table` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `dt` datetime DEFAULT NULL,
  `dth` varchar(13) DEFAULT NULL,
  `Action_cycle_of_main_expansion_valve` int(11) DEFAULT NULL,
  `Ambient_temperature` float DEFAULT NULL,
  `Compressor_actual_frequency` int(11) DEFAULT NULL,
  `Compressor_current` float DEFAULT NULL,
  `Defrost_cycle` int(11) DEFAULT NULL,
  `Defrost_frequency` int(11) DEFAULT NULL,
  `Defrost_time` int(11) DEFAULT NULL,
  `Exhaust_Gas_temperature` float DEFAULT NULL,
  `External_coil_temperature` float DEFAULT NULL,
  `Inlet_water_temperature` float DEFAULT NULL,
  `Inner_coil_temperature` float DEFAULT NULL,
  `Low_pressure_conversion_temperature` float DEFAULT NULL,
  `Low_pressure_value` float DEFAULT NULL,
  `Outlet_water_temperature` float DEFAULT NULL,
  `Suction_gas_temperature` float DEFAULT NULL,
  `Water_tank_temperature` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `dt` (`dt`)
) ENGINE=InnoDB AUTO_INCREMENT=65422 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pv_easun`
--

DROP TABLE IF EXISTS `pv_easun`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `pv_easun` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `timestamp` datetime DEFAULT NULL,
  `grid_voltage` float DEFAULT NULL,
  `grid_frequency` float DEFAULT NULL,
  `ac_output_voltage` float DEFAULT NULL,
  `ac_output_frequency` float DEFAULT NULL,
  `ac_output_apparent_power` float DEFAULT NULL,
  `ac_output_active_power` float DEFAULT NULL,
  `output_load_percent` float DEFAULT NULL,
  `battery_voltage` float DEFAULT NULL,
  `battery_discharge_current` float DEFAULT NULL,
  `battery_charging_current` float DEFAULT NULL,
  `battery_capacity` float DEFAULT NULL,
  `inverter_heat_sink_temperature` float DEFAULT NULL,
  `pv1_input_power` float DEFAULT NULL,
  `pv2_input_power` float DEFAULT NULL,
  `pv1_input_voltage` float DEFAULT NULL,
  `pv2_input_voltage` float DEFAULT NULL,
  `efficiency` float DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=40018 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pv_ebox`
--

DROP TABLE IF EXISTS `pv_ebox`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `pv_ebox` (
  `id` mediumint(9) NOT NULL AUTO_INCREMENT,
  `Power` int(11) DEFAULT NULL,
  `Volt` int(11) DEFAULT NULL,
  `Curr` int(11) DEFAULT NULL,
  `Tempr` int(11) DEFAULT NULL,
  `Tlow` int(11) DEFAULT NULL,
  `Thigh` int(11) DEFAULT NULL,
  `Vlow` int(11) DEFAULT NULL,
  `Vhigh` int(11) DEFAULT NULL,
  `BaseSt` varchar(9) DEFAULT NULL,
  `VoltSt` varchar(9) DEFAULT NULL,
  `CurrSt` varchar(9) DEFAULT NULL,
  `TempSt` varchar(9) DEFAULT NULL,
  `Coulomb` int(11) DEFAULT NULL,
  `ts` varchar(19) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=178684 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pv_ebox2`
--

DROP TABLE IF EXISTS `pv_ebox2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `pv_ebox2` (
  `id` mediumint(9) NOT NULL AUTO_INCREMENT,
  `Power` int(11) DEFAULT NULL,
  `Volt` int(11) DEFAULT NULL,
  `Curr` int(11) DEFAULT NULL,
  `Tempr` int(11) DEFAULT NULL,
  `Tlow` int(11) DEFAULT NULL,
  `Thigh` int(11) DEFAULT NULL,
  `Vlow` int(11) DEFAULT NULL,
  `Vhigh` int(11) DEFAULT NULL,
  `BaseSt` varchar(9) DEFAULT NULL,
  `VoltSt` varchar(9) DEFAULT NULL,
  `CurrSt` varchar(9) DEFAULT NULL,
  `TempSt` varchar(9) DEFAULT NULL,
  `Coulomb` int(11) DEFAULT NULL,
  `ts` datetime DEFAULT NULL,
  `ts_min` datetime GENERATED ALWAYS AS ('2024-01-01' + interval timestampdiff(MINUTE,'2024-01-01',`ts`) minute) VIRTUAL,
  PRIMARY KEY (`id`),
  KEY `pv_ebox2_ts_IDX` (`ts`) USING BTREE,
  KEY `ts_min` (`ts_min`)
) ENGINE=InnoDB AUTO_INCREMENT=4545642 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pv_em2`
--

DROP TABLE IF EXISTS `pv_em2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `pv_em2` (
  `id` mediumint(9) NOT NULL AUTO_INCREMENT,
  `dt` datetime DEFAULT NULL,
  `c2` float DEFAULT NULL,
  `c4` float DEFAULT NULL,
  `c6` float DEFAULT NULL,
  `c8` float DEFAULT NULL,
  `c10` float DEFAULT NULL,
  `c12` float DEFAULT NULL,
  `c14` float DEFAULT NULL,
  `c16` float DEFAULT NULL,
  `c18` float DEFAULT NULL,
  `c20` float DEFAULT NULL,
  `c22` float DEFAULT NULL,
  `c24` float DEFAULT NULL,
  `c26` float DEFAULT NULL,
  `c28` float DEFAULT NULL,
  `c30` float DEFAULT NULL,
  `c32` float DEFAULT NULL,
  `c34` float DEFAULT NULL,
  `c36` float DEFAULT NULL,
  `c44` float DEFAULT NULL,
  `c48` float DEFAULT NULL,
  `c50` float DEFAULT NULL,
  `c54` float DEFAULT NULL,
  `c58` float DEFAULT NULL,
  `c62` float DEFAULT NULL,
  `c64` float DEFAULT NULL,
  `c72` float DEFAULT NULL,
  `c74` float DEFAULT NULL,
  `c76` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `pv_em2_dt_IDX` (`dt`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=664153 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pv_emsec`
--

DROP TABLE IF EXISTS `pv_emsec`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `pv_emsec` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `dt` datetime DEFAULT NULL,
  `c2` float DEFAULT NULL,
  `c4` float DEFAULT NULL,
  `c6` float DEFAULT NULL,
  `c8` float DEFAULT NULL,
  `c10` float DEFAULT NULL,
  `c12` float DEFAULT NULL,
  `c14` float DEFAULT NULL,
  `c16` float DEFAULT NULL,
  `c18` float DEFAULT NULL,
  `c20` float DEFAULT NULL,
  `c22` float DEFAULT NULL,
  `c24` float DEFAULT NULL,
  `c26` float DEFAULT NULL,
  `c28` float DEFAULT NULL,
  `c30` float DEFAULT NULL,
  `c32` float DEFAULT NULL,
  `c34` float DEFAULT NULL,
  `c36` float DEFAULT NULL,
  `c44` float DEFAULT NULL,
  `c48` float DEFAULT NULL,
  `c50` float DEFAULT NULL,
  `c54` float DEFAULT NULL,
  `c58` float DEFAULT NULL,
  `c62` float DEFAULT NULL,
  `c64` float DEFAULT NULL,
  `c72` float DEFAULT NULL,
  `c74` float DEFAULT NULL,
  `c76` float DEFAULT NULL,
  `dth` varchar(13) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `pv_emsec_dt_IDX` (`dt`) USING BTREE,
  KEY `i_dth` (`dth`),
  KEY `idx_pv_emsec_dth` (`dth`)
) ENGINE=InnoDB AUTO_INCREMENT=15430345 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pv_lastprofil`
--

DROP TABLE IF EXISTS `pv_lastprofil`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `pv_lastprofil` (
  `datum` varchar(10) DEFAULT NULL,
  `zeit` varchar(8) DEFAULT NULL,
  `leistung` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pv_lastprofil1`
--

DROP TABLE IF EXISTS `pv_lastprofil1`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `pv_lastprofil1` (
  `id` mediumint(9) NOT NULL AUTO_INCREMENT,
  `datum` varchar(10) NOT NULL,
  `zeit` varchar(8) NOT NULL,
  `leistung` float DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=167 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pv_lastprofil2`
--

DROP TABLE IF EXISTS `pv_lastprofil2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `pv_lastprofil2` (
  `id` mediumint(9) NOT NULL AUTO_INCREMENT,
  `datum` varchar(10) NOT NULL,
  `zeit` varchar(8) NOT NULL,
  `leistung` float DEFAULT NULL,
  `einspeis` float DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1003544 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pv_zaehl`
--

DROP TABLE IF EXISTS `pv_zaehl`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `pv_zaehl` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `datum` varchar(19) DEFAULT NULL,
  `bezug` decimal(10,4) DEFAULT NULL,
  `einspeis` decimal(10,4) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=56206 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary table structure for view `pv_zaehl00`
--

DROP TABLE IF EXISTS `pv_zaehl00`;
/*!50001 DROP VIEW IF EXISTS `pv_zaehl00`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8mb4;
/*!50001 CREATE VIEW `pv_zaehl00` AS SELECT
 1 AS `datum`,
  1 AS `bezug`,
  1 AS `einspeis` */;
SET character_set_client = @saved_cs_client;

--
-- Temporary table structure for view `pv_zaehl08`
--

DROP TABLE IF EXISTS `pv_zaehl08`;
/*!50001 DROP VIEW IF EXISTS `pv_zaehl08`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8mb4;
/*!50001 CREATE VIEW `pv_zaehl08` AS SELECT
 1 AS `datum`,
  1 AS `bezug`,
  1 AS `einspeis` */;
SET character_set_client = @saved_cs_client;

--
-- Temporary table structure for view `pv_zaehl10`
--

DROP TABLE IF EXISTS `pv_zaehl10`;
/*!50001 DROP VIEW IF EXISTS `pv_zaehl10`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8mb4;
/*!50001 CREATE VIEW `pv_zaehl10` AS SELECT
 1 AS `datum`,
  1 AS `bezug`,
  1 AS `einspeis` */;
SET character_set_client = @saved_cs_client;

--
-- Temporary table structure for view `pv_zaehl19`
--

DROP TABLE IF EXISTS `pv_zaehl19`;
/*!50001 DROP VIEW IF EXISTS `pv_zaehl19`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8mb4;
/*!50001 CREATE VIEW `pv_zaehl19` AS SELECT
 1 AS `datum`,
  1 AS `bezug`,
  1 AS `einspeis` */;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `pv_zaehl2`
--

DROP TABLE IF EXISTS `pv_zaehl2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `pv_zaehl2` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `datum` datetime DEFAULT NULL,
  `bezug` decimal(10,4) DEFAULT NULL,
  `einspeis` decimal(10,4) DEFAULT NULL,
  `wirkleist` decimal(10,4) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `pv_zaehl2_datum_IDX` (`datum`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=1607732 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pve_events`
--

DROP TABLE IF EXISTS `pve_events`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `pve_events` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `event_time` timestamp NULL DEFAULT current_timestamp(),
  `device` varchar(50) DEFAULT NULL,
  `status` varchar(500) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=24 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sdm72d`
--

DROP TABLE IF EXISTS `sdm72d`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `sdm72d` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `timestamp` datetime NOT NULL,
  `voltage_l1` float DEFAULT NULL,
  `voltage_l2` float DEFAULT NULL,
  `current_l1` float DEFAULT NULL,
  `current_l2` float DEFAULT NULL,
  `current_l3` float DEFAULT NULL,
  `active_power_l1` float DEFAULT NULL,
  `active_power_l2` float DEFAULT NULL,
  `active_power_l3` float DEFAULT NULL,
  `apparent_power_l1` float DEFAULT NULL,
  `apparent_power_l2` float DEFAULT NULL,
  `apparent_power_l3` float DEFAULT NULL,
  `reactive_power_l1` float DEFAULT NULL,
  `reactive_power_l2` float DEFAULT NULL,
  `reactive_power_l3` float DEFAULT NULL,
  `power_factor_l1` float DEFAULT NULL,
  `power_factor_l2` float DEFAULT NULL,
  `power_factor_l3` float DEFAULT NULL,
  `avg_voltage_ln` float DEFAULT NULL,
  `avg_current` float DEFAULT NULL,
  `sum_current` float DEFAULT NULL,
  `total_active_power` float DEFAULT NULL,
  `total_apparent_power` float DEFAULT NULL,
  `total_reactive_power` float DEFAULT NULL,
  `total_power_factor` float DEFAULT NULL,
  `frequency` float DEFAULT NULL,
  `total_import_active_energy` float DEFAULT NULL,
  `total_export_active_energy` float DEFAULT NULL,
  `voltage_l1_l2` float DEFAULT NULL,
  `voltage_l2_l3` float DEFAULT NULL,
  `voltage_l3_l1` float DEFAULT NULL,
  `avg_voltage_ll` float DEFAULT NULL,
  `neutral_current` float DEFAULT NULL,
  `total_active_energy` float DEFAULT NULL,
  `total_reactive_energy` float DEFAULT NULL,
  `resettable_total_active_energy` float DEFAULT NULL,
  `resettable_total_reactive_energy` float DEFAULT NULL,
  `resettable_import_active_energy` float DEFAULT NULL,
  `resettable_export_active_energy` float DEFAULT NULL,
  `net_kwh` float DEFAULT NULL,
  `total_import_active_power` float DEFAULT NULL,
  `total_export_active_power` float DEFAULT NULL,
  `hour` varchar(16) GENERATED ALWAYS AS (replace(left(`timestamp`,13),' ','-')) STORED,
  PRIMARY KEY (`id`),
  KEY `idx_sdm72d_timestamp` (`timestamp`),
  KEY `idx_hour` (`hour`)
) ENGINE=InnoDB AUTO_INCREMENT=123092 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sofar_inverter_data`
--

DROP TABLE IF EXISTS `sofar_inverter_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `sofar_inverter_data` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `timestamp` datetime DEFAULT current_timestamp(),
  `register` varchar(6) NOT NULL,
  `name` varchar(255) DEFAULT NULL,
  `value` decimal(15,5) DEFAULT NULL,
  `unit` varchar(10) DEFAULT NULL,
  `type` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_register` (`register`),
  KEY `idx_timestamp` (`timestamp`)
) ENGINE=InnoDB AUTO_INCREMENT=7900 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sofar_pivot`
--

DROP TABLE IF EXISTS `sofar_pivot`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `sofar_pivot` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `timestamp` datetime DEFAULT current_timestamp(),
  `voltage_pv1` decimal(10,2) DEFAULT NULL,
  `current_pv1` decimal(10,2) DEFAULT NULL,
  `power_pv1` decimal(10,2) DEFAULT NULL,
  `voltage_pv2` decimal(10,2) DEFAULT NULL,
  `current_pv2` decimal(10,2) DEFAULT NULL,
  `power_pv2` decimal(10,2) DEFAULT NULL,
  `power_pv_total` decimal(10,2) DEFAULT NULL,
  `voltage_phase_r` decimal(10,2) DEFAULT NULL,
  `current_phase_r` decimal(10,2) DEFAULT NULL,
  `voltage_phase_s` decimal(10,2) DEFAULT NULL,
  `current_phase_s` decimal(10,2) DEFAULT NULL,
  `voltage_phase_t` decimal(10,2) DEFAULT NULL,
  `current_phase_t` decimal(10,2) DEFAULT NULL,
  `active_power_output_total` decimal(10,2) DEFAULT NULL,
  `active_power_pcc_total` decimal(10,2) DEFAULT NULL,
  `reactive_power_output_total` decimal(10,2) DEFAULT NULL,
  `reactive_power_pcc_total` decimal(10,2) DEFAULT NULL,
  `daily_energy` decimal(10,2) DEFAULT NULL,
  `total_energy` decimal(15,2) DEFAULT NULL,
  `frequency_grid` decimal(5,2) DEFAULT NULL,
  `temperature_inverter` decimal(5,2) DEFAULT NULL,
  `temperature_boost` decimal(5,2) DEFAULT NULL,
  `temperature_env` decimal(5,2) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_timestamp` (`timestamp`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sofar_pivot2`
--

DROP TABLE IF EXISTS `sofar_pivot2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `sofar_pivot2` (
  `timestamp` datetime NOT NULL,
  `AddressMask_General1` bigint(20) unsigned DEFAULT NULL,
  `Modbus_Protocol_Version` int(10) unsigned DEFAULT NULL,
  `COMM_MCU_Code` varchar(50) DEFAULT NULL,
  `CTRL1_MCU_Code` varchar(50) DEFAULT NULL,
  `CTRL2_MCU_Code` varchar(50) DEFAULT NULL,
  `Protocol_ARM_DSPM` int(10) unsigned DEFAULT NULL,
  `Protocol_ARM_DSPS` int(10) unsigned DEFAULT NULL,
  `Protocol_DSPM_DSPS` int(10) unsigned DEFAULT NULL,
  `SysState` int(10) unsigned DEFAULT NULL,
  `Countdown` int(10) unsigned DEFAULT NULL,
  `Temperature_Env1` smallint(6) DEFAULT NULL,
  `Temperature_HeatSink1` smallint(6) DEFAULT NULL,
  `Temperature_Inv1` smallint(6) DEFAULT NULL,
  `GenerationTime_Today` int(10) unsigned DEFAULT NULL,
  `GenerationTime_Total` int(10) unsigned DEFAULT NULL,
  `ServiceTime_Total` int(10) unsigned DEFAULT NULL,
  `InsulationResistance` int(10) unsigned DEFAULT NULL,
  `SysTime_Year` int(10) unsigned DEFAULT NULL,
  `SysTime_Month` int(10) unsigned DEFAULT NULL,
  `SysTime_Date` int(10) unsigned DEFAULT NULL,
  `SysTime_Hour` int(10) unsigned DEFAULT NULL,
  `SysTime_Minute` int(10) unsigned DEFAULT NULL,
  `SysTime_Second` int(10) unsigned DEFAULT NULL,
  `Production_Code` int(10) unsigned DEFAULT NULL,
  `Serial_Number` varchar(50) DEFAULT NULL,
  `Hardware_Version` varchar(50) DEFAULT NULL,
  `Software_Version_Stage_COM` varchar(10) DEFAULT NULL,
  `Software_Version_Major_COM` varchar(10) DEFAULT NULL,
  `Software_Version_Custom_COM` varchar(10) DEFAULT NULL,
  `Software_Version_Minor_COM` varchar(10) DEFAULT NULL,
  `Software_Version_Stage_Master` varchar(10) DEFAULT NULL,
  `Software_Version_Major_Master` varchar(10) DEFAULT NULL,
  `Software_Version_Custom_Master` varchar(10) DEFAULT NULL,
  `Software_Version_Minor_Master` varchar(10) DEFAULT NULL,
  `Software_Version_Stage_Slave` varchar(10) DEFAULT NULL,
  `Software_Version_Major_Slave` varchar(10) DEFAULT NULL,
  `Software_Version_Custom_Slave` varchar(10) DEFAULT NULL,
  `Software_Version_Minor_Slave` varchar(10) DEFAULT NULL,
  `Safety_Version_Major` int(10) unsigned DEFAULT NULL,
  `Safety_version_Minor` int(10) unsigned DEFAULT NULL,
  `Boot_Version_COM` int(10) unsigned DEFAULT NULL,
  `Boot_Version_Master` int(10) unsigned DEFAULT NULL,
  `Boot_Version_Slave` int(10) unsigned DEFAULT NULL,
  `Safety_Firmware_Version_Stage` varchar(10) DEFAULT NULL,
  `Safety_Firmware_Version_Major` varchar(10) DEFAULT NULL,
  `Safety_Firmware_Version_Custom` varchar(10) DEFAULT NULL,
  `Safety_Firmware_Version_Minor` varchar(10) DEFAULT NULL,
  `Safety_Hardware_Version` varchar(50) DEFAULT NULL,
  `Afci_Ver_Firmware` varchar(10) DEFAULT NULL,
  `Afci_Ver_Sft` varchar(10) DEFAULT NULL,
  `Afci_Ver_Hard` varchar(10) DEFAULT NULL,
  `Afci_Srs_Hard` varchar(10) DEFAULT NULL,
  `Afci_Ver_Com` varchar(10) DEFAULT NULL,
  `Safety_Package_Version` varchar(50) DEFAULT NULL,
  `Frequency_Grid` decimal(5,2) DEFAULT NULL,
  `ActivePower_Output_Total` decimal(7,2) DEFAULT NULL,
  `ReactivePower_Output_Total` decimal(7,2) DEFAULT NULL,
  `ApparentPower_Output_Total` decimal(7,2) DEFAULT NULL,
  `ActivePower_PCC_Total` decimal(7,2) DEFAULT NULL,
  `ReactivePower_PCC_Total` decimal(7,2) DEFAULT NULL,
  `ApparentPower_PCC_Total` decimal(7,2) DEFAULT NULL,
  `Voltage_Phase_R` decimal(5,1) DEFAULT NULL,
  `Current_Output_R` decimal(5,2) DEFAULT NULL,
  `ActivePower_Output_R` decimal(7,2) DEFAULT NULL,
  `ReactivePower_Output_R` decimal(7,2) DEFAULT NULL,
  `PowerFactor_Output_R` decimal(5,3) DEFAULT NULL,
  `Current_PCC_R` decimal(5,2) DEFAULT NULL,
  `ActivePower_PCC_R` decimal(7,2) DEFAULT NULL,
  `ReactivePower_PCC_R` decimal(7,2) DEFAULT NULL,
  `PowerFactor_PCC_R` decimal(5,3) DEFAULT NULL,
  `Voltage_Phase_S` decimal(5,1) DEFAULT NULL,
  `Current_Output_S` decimal(5,2) DEFAULT NULL,
  `ActivePower_Output_S` decimal(7,2) DEFAULT NULL,
  `ReactivePower_Output_S` decimal(7,2) DEFAULT NULL,
  `PowerFactor_Output_S` decimal(5,3) DEFAULT NULL,
  `Current_PCC_S` decimal(5,2) DEFAULT NULL,
  `ActivePower_PCC_S` decimal(7,2) DEFAULT NULL,
  `ReactivePower_PCC_S` decimal(7,2) DEFAULT NULL,
  `PowerFactor_PCC_S` decimal(5,3) DEFAULT NULL,
  `Voltage_Phase_T` decimal(5,1) DEFAULT NULL,
  `Current_Output_T` decimal(5,2) DEFAULT NULL,
  `ActivePower_Output_T` decimal(7,2) DEFAULT NULL,
  `ReactivePower_Output_T` decimal(7,2) DEFAULT NULL,
  `PowerFactor_Output_T` decimal(5,3) DEFAULT NULL,
  `Current_PCC_T` decimal(5,2) DEFAULT NULL,
  `ActivePower_PCC_T` decimal(7,2) DEFAULT NULL,
  `ReactivePower_PCC_T` decimal(7,2) DEFAULT NULL,
  `PowerFactor_PCC_T` decimal(5,3) DEFAULT NULL,
  `ActivePower_PV_Ext` decimal(7,2) DEFAULT NULL,
  `ActivePower_Load_Sys` decimal(7,2) DEFAULT NULL,
  `Voltage_Phase_L1N` decimal(5,1) DEFAULT NULL,
  `Current_Output_L1N` decimal(5,2) DEFAULT NULL,
  `ActivePower_Output_L1N` decimal(7,2) DEFAULT NULL,
  `Current_PCC_L1N` decimal(5,2) DEFAULT NULL,
  `ActivePower_PCC_L1N` decimal(7,2) DEFAULT NULL,
  `Voltage_Phase_L2N` decimal(5,1) DEFAULT NULL,
  `Current_Output_L2N` decimal(5,2) DEFAULT NULL,
  `ActivePower_Output_L2N` decimal(7,2) DEFAULT NULL,
  `Current_PCC_L2N` decimal(5,2) DEFAULT NULL,
  `ActivePower_PCC_L2N` decimal(7,2) DEFAULT NULL,
  `Voltage_Line_L1` decimal(5,1) DEFAULT NULL,
  `Voltage_Line_L2` decimal(5,1) DEFAULT NULL,
  `Voltage_Line_L3` decimal(5,1) DEFAULT NULL,
  `Power_Factor` decimal(5,3) DEFAULT NULL,
  `ActivePower_Load_Total` decimal(7,2) DEFAULT NULL,
  `ReactivePower_Load_Total` decimal(7,2) DEFAULT NULL,
  `ApparentPower_Load_Total` decimal(7,2) DEFAULT NULL,
  `Frequency_Output` decimal(5,2) DEFAULT NULL,
  `Voltage_Output_R` decimal(5,1) DEFAULT NULL,
  `Current_Load_R` decimal(5,2) DEFAULT NULL,
  `ActivePower_Load_R` decimal(7,2) DEFAULT NULL,
  `ReactivePower_Load_R` decimal(7,2) DEFAULT NULL,
  `ApparentPower_Load_R` decimal(7,2) DEFAULT NULL,
  `LoadPeakRatio_R` decimal(5,2) DEFAULT NULL,
  `Voltage_Load_R` decimal(5,1) DEFAULT NULL,
  `Voltage_Output_S` decimal(5,1) DEFAULT NULL,
  `Current_Load_S` decimal(5,2) DEFAULT NULL,
  `ActivePower_Load_S` decimal(7,2) DEFAULT NULL,
  `ReactivePower_Load_S` decimal(7,2) DEFAULT NULL,
  `ApparentPower_Load_S` decimal(7,2) DEFAULT NULL,
  `LoadPeakRatio_S` decimal(5,2) DEFAULT NULL,
  `Voltage_Load_S` decimal(5,1) DEFAULT NULL,
  `Voltage_Output_T` decimal(5,1) DEFAULT NULL,
  `Current_Load_T` decimal(5,2) DEFAULT NULL,
  `ActivePower_Load_T` decimal(7,2) DEFAULT NULL,
  `ReactivePower_Load_T` decimal(7,2) DEFAULT NULL,
  `ApparentPower_Load_T` decimal(7,2) DEFAULT NULL,
  `LoadPeakRatio_T` decimal(5,2) DEFAULT NULL,
  `Voltage_Load_T` decimal(5,1) DEFAULT NULL,
  `Voltage_Output_L1N` decimal(5,1) DEFAULT NULL,
  `Current_Load_L1N` decimal(5,2) DEFAULT NULL,
  `ActivePower_Load_L1N` decimal(7,2) DEFAULT NULL,
  `Voltage_Output_L2N` decimal(5,1) DEFAULT NULL,
  `Current_Load_L2N` decimal(5,2) DEFAULT NULL,
  `ActivePower_Load_L2N` decimal(7,2) DEFAULT NULL,
  `Voltage_PV1` decimal(5,1) DEFAULT NULL,
  `Current_PV1` decimal(5,2) DEFAULT NULL,
  `Power_PV1` decimal(7,2) DEFAULT NULL,
  `Power_PV_Total` decimal(7,1) DEFAULT NULL,
  `Voltage_Bat1` decimal(5,1) DEFAULT NULL,
  `Current_Bat1` decimal(5,2) DEFAULT NULL,
  `Power_Bat1` decimal(7,2) DEFAULT NULL,
  `Temperature_Env_Bat1` smallint(6) DEFAULT NULL,
  `SOC_Bat1` tinyint(3) unsigned DEFAULT NULL,
  `SOH_Bat1` tinyint(3) unsigned DEFAULT NULL,
  `ChargeCycle_Bat1` int(10) unsigned DEFAULT NULL,
  `Power_Bat_Total` decimal(7,1) DEFAULT NULL,
  `SOC_Bat_Average` tinyint(3) unsigned DEFAULT NULL,
  `SOH_Bat` tinyint(3) unsigned DEFAULT NULL,
  `CurrentBattery_num` tinyint(3) unsigned DEFAULT NULL,
  `PV_Generation_Today` decimal(8,2) DEFAULT NULL,
  `PV_Generation_Total` decimal(10,1) DEFAULT NULL,
  `Load_Consumption_Today` decimal(8,2) DEFAULT NULL,
  `Load_Consumption_Total` decimal(10,1) DEFAULT NULL,
  `Energy_Purchase_Today` decimal(8,2) DEFAULT NULL,
  `Energy_Purchase_Total` decimal(10,1) DEFAULT NULL,
  `Energy_Selling_Today` decimal(8,2) DEFAULT NULL,
  `Energy_Selling_Total` decimal(10,1) DEFAULT NULL,
  `Bat_Charge_Today` decimal(8,2) DEFAULT NULL,
  `Bat_Charge_Total` decimal(10,1) DEFAULT NULL,
  `Bat_Discharge_Today` decimal(8,2) DEFAULT NULL,
  `Bat_Discharge_Total` decimal(10,1) DEFAULT NULL,
  `PV_Generation_Month` decimal(10,1) DEFAULT NULL,
  `PV_Generation_Year` decimal(10,1) DEFAULT NULL,
  `Load_Consumption_Month` decimal(10,1) DEFAULT NULL,
  `Load_Consumption_Year` decimal(10,1) DEFAULT NULL,
  `Energy_Purchase_Month` decimal(10,1) DEFAULT NULL,
  `Energy_Purchase_Year` decimal(10,1) DEFAULT NULL,
  `Energy_Selling_Month` decimal(10,1) DEFAULT NULL,
  `Energy_Selling_Year` decimal(10,1) DEFAULT NULL,
  `Bat_Charge_Month` decimal(10,1) DEFAULT NULL,
  `Bat_Charge_Year` decimal(10,1) DEFAULT NULL,
  `Bat_Discharge_Month` decimal(10,1) DEFAULT NULL,
  `Bat_Discharge_Year` decimal(10,1) DEFAULT NULL,
  `GFCI` int(10) unsigned DEFAULT NULL,
  `Current_Bus_Balance` decimal(5,2) DEFAULT NULL,
  `DCI_R` smallint(6) DEFAULT NULL,
  `DCI_S` smallint(6) DEFAULT NULL,
  `DCI_T` smallint(6) DEFAULT NULL,
  `DCV_R` smallint(6) DEFAULT NULL,
  `DCV_S` smallint(6) DEFAULT NULL,
  `DCV_T` smallint(6) DEFAULT NULL,
  `Voltage_Bus` decimal(5,1) DEFAULT NULL,
  `Voltage_Bus_P` decimal(5,1) DEFAULT NULL,
  `Voltage_Bus_N` decimal(5,1) DEFAULT NULL,
  `Voltage_Bus_LLC` decimal(5,1) DEFAULT NULL,
  `Current_BuckBoost` decimal(5,2) DEFAULT NULL,
  `Voltage_Bus_P_Half` decimal(5,1) DEFAULT NULL,
  `Voltage_Bus_N_Half` decimal(5,1) DEFAULT NULL,
  `FlyingCap_Voltage1` decimal(5,1) DEFAULT NULL,
  `RatedPower_Inverter` decimal(5,1) DEFAULT NULL,
  `IVPoints_Effec_Num` int(10) unsigned DEFAULT NULL,
  `Voltage_Group1` decimal(5,1) DEFAULT NULL,
  `Current_Group1_Branch1` decimal(5,2) DEFAULT NULL,
  `Current_Group1_Branch2` decimal(5,2) DEFAULT NULL,
  `Current_Group1_Branch3` decimal(5,2) DEFAULT NULL,
  `Current_Group1_Branch4` decimal(5,2) DEFAULT NULL,
  `ArcStrength_MPPT_1` smallint(6) DEFAULT NULL,
  `ArcStrength_history_Channel1` smallint(6) DEFAULT NULL,
  `arc_peak_to_peak` int(10) unsigned DEFAULT NULL,
  `arc_variance` int(10) unsigned DEFAULT NULL,
  `arc_harmonic_energy` int(10) unsigned DEFAULT NULL,
  `arc_amplitude_variance` int(10) unsigned DEFAULT NULL,
  PRIMARY KEY (`timestamp`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sofar_registers`
--

DROP TABLE IF EXISTS `sofar_registers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `sofar_registers` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `function_name` varchar(255) DEFAULT NULL,
  `register_address` varchar(20) DEFAULT NULL,
  `field_name` varchar(255) DEFAULT NULL,
  `data_type` varchar(20) DEFAULT NULL,
  `accuracy` varchar(20) DEFAULT NULL,
  `units` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `strom`
--

DROP TABLE IF EXISTS `strom`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `strom` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `epoch` bigint(20) NOT NULL,
  `serverid` bigint(20) NOT NULL,
  `HT` decimal(10,2) DEFAULT NULL,
  `NT` decimal(10,2) DEFAULT NULL,
  `DELIVER` decimal(10,2) DEFAULT NULL,
  `current` decimal(10,2) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=4465 DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `synset`
--

DROP TABLE IF EXISTS `synset`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `synset` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `version` bigint(20) NOT NULL,
  `evaluation` int(11) DEFAULT NULL,
  `is_visible` bit(1) NOT NULL,
  `original_id` int(11) DEFAULT NULL,
  `preferred_category_id` bigint(20) DEFAULT NULL,
  `section_id` bigint(20) DEFAULT NULL,
  `source_id` bigint(20) DEFAULT NULL,
  `synset_preferred_term` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `FKCB1A289A4A27BB5` (`source_id`),
  KEY `FKCB1A289AED375357` (`preferred_category_id`),
  KEY `FKCB1A289AD9BB831F` (`section_id`),
  KEY `is_visible` (`is_visible`),
  KEY `original_id` (`original_id`)
) ENGINE=MyISAM AUTO_INCREMENT=35872 DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `synset_link`
--

DROP TABLE IF EXISTS `synset_link`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `synset_link` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `version` bigint(20) NOT NULL,
  `evaluation_status` int(11) DEFAULT NULL,
  `fact_count` int(11) DEFAULT NULL,
  `link_type_id` bigint(20) NOT NULL,
  `synset_id` bigint(20) NOT NULL,
  `target_synset_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `FK6336907FA3843755` (`synset_id`),
  KEY `FK6336907FE4B8FC6A` (`link_type_id`),
  KEY `FK6336907F9933A227` (`target_synset_id`)
) ENGINE=MyISAM AUTO_INCREMENT=30813 DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tbl_mq`
--

DROP TABLE IF EXISTS `tbl_mq`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `tbl_mq` (
  `messageID` int(11) NOT NULL AUTO_INCREMENT,
  `clientID` varchar(20) NOT NULL,
  `topic` varchar(50) NOT NULL,
  `message` varchar(100) NOT NULL,
  `Enable` tinyint(1) DEFAULT 1,
  `DateTime_created` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`messageID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tcp_archerc7`
--

DROP TABLE IF EXISTS `tcp_archerc7`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `tcp_archerc7` (
  `ddate` varchar(16) DEFAULT NULL,
  `dtime` varchar(16) DEFAULT NULL,
  `ipver` varchar(3) DEFAULT NULL,
  `src` varchar(66) DEFAULT NULL,
  `dest` varchar(66) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `term`
--

DROP TABLE IF EXISTS `term`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `term` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `version` bigint(20) NOT NULL,
  `language_id` bigint(20) NOT NULL,
  `level_id` bigint(20) DEFAULT NULL,
  `normalized_word` varchar(255) CHARACTER SET utf8mb3 COLLATE utf8mb3_general_ci DEFAULT NULL,
  `original_id` int(11) DEFAULT NULL,
  `synset_id` bigint(20) NOT NULL,
  `user_comment` varchar(255) CHARACTER SET utf8mb3 COLLATE utf8mb3_general_ci DEFAULT NULL,
  `word` varchar(255) CHARACTER SET utf8mb3 COLLATE utf8mb3_general_ci NOT NULL,
  `normalized_word2` varchar(255) CHARACTER SET utf8mb3 COLLATE utf8mb3_general_ci DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `FK36446CA3843755` (`synset_id`),
  KEY `FK36446C534B2C73` (`level_id`),
  KEY `FK36446C5CA8CBD5` (`language_id`),
  KEY `word` (`word`),
  KEY `normalized_word` (`normalized_word`),
  KEY `normalized_word2` (`normalized_word2`)
) ENGINE=MyISAM AUTO_INCREMENT=144237 DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `term_level`
--

DROP TABLE IF EXISTS `term_level`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `term_level` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `version` bigint(20) NOT NULL,
  `level_name` varchar(255) NOT NULL,
  `short_level_name` varchar(255) NOT NULL,
  `sort_value` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `level_name` (`level_name`),
  UNIQUE KEY `short_level_name` (`short_level_name`)
) ENGINE=MyISAM AUTO_INCREMENT=7 DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `term_link`
--

DROP TABLE IF EXISTS `term_link`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `term_link` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `version` bigint(20) NOT NULL,
  `link_type_id` bigint(20) NOT NULL,
  `target_term_id` bigint(20) NOT NULL,
  `term_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `FK78CD07EDDBDC7376` (`link_type_id`),
  KEY `FK78CD07ED36F08D5` (`term_id`),
  KEY `FK78CD07ED2D3F0027` (`target_term_id`)
) ENGINE=MyISAM AUTO_INCREMENT=2162 DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `term_link_type`
--

DROP TABLE IF EXISTS `term_link_type`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `term_link_type` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `version` bigint(20) NOT NULL,
  `link_name` varchar(255) NOT NULL,
  `other_direction_link_name` varchar(255) NOT NULL,
  `verb_name` varchar(255) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `link_name` (`link_name`),
  UNIQUE KEY `verb_name` (`verb_name`)
) ENGINE=MyISAM AUTO_INCREMENT=2 DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `term_tag`
--

DROP TABLE IF EXISTS `term_tag`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `term_tag` (
  `term_tags_id` bigint(20) DEFAULT NULL,
  `tag_id` bigint(20) DEFAULT NULL,
  KEY `FKB9931D47C610557F` (`tag_id`),
  KEY `FKB9931D47AD241D35` (`term_tags_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `transaction_inputs`
--

DROP TABLE IF EXISTS `transaction_inputs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `transaction_inputs` (
  `transaction_hash` varchar(64) NOT NULL,
  `index` int(11) NOT NULL,
  `spent_transaction_hash` varchar(64) DEFAULT NULL,
  `spent_output_index` int(11) DEFAULT NULL,
  `script_asm` longtext DEFAULT NULL,
  `script_hex` longtext DEFAULT NULL,
  `sequence` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`transaction_hash`,`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `transaction_outputs`
--

DROP TABLE IF EXISTS `transaction_outputs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `transaction_outputs` (
  `transaction_hash` varchar(64) NOT NULL,
  `index` int(11) NOT NULL,
  `script_asm` longtext DEFAULT NULL,
  `script_hex` longtext DEFAULT NULL,
  `required_signatures` int(11) DEFAULT NULL,
  `type` varchar(100) DEFAULT NULL,
  `addresses` longtext DEFAULT NULL,
  `value` decimal(20,8) DEFAULT NULL,
  PRIMARY KEY (`transaction_hash`,`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `transactions`
--

DROP TABLE IF EXISTS `transactions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `transactions` (
  `hash` varchar(64) NOT NULL,
  `nonce` bigint(20) DEFAULT NULL,
  `block_hash` varchar(64) DEFAULT NULL,
  `block_number` bigint(20) DEFAULT NULL,
  `transaction_index` bigint(20) DEFAULT NULL,
  `from_address` varchar(42) DEFAULT NULL,
  `to_address` varchar(42) DEFAULT NULL,
  `value` decimal(38,0) DEFAULT NULL,
  `gas` bigint(20) DEFAULT NULL,
  `gas_price` decimal(38,0) DEFAULT NULL,
  `input` longtext DEFAULT NULL,
  `block_timestamp` bigint(20) DEFAULT NULL,
  `max_fee_per_gas` decimal(38,0) DEFAULT NULL,
  `max_priority_fee_per_gas` decimal(38,0) DEFAULT NULL,
  `transaction_type` int(11) DEFAULT NULL,
  PRIMARY KEY (`hash`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary table structure for view `v01inverter_ebox`
--

DROP TABLE IF EXISTS `v01inverter_ebox`;
/*!50001 DROP VIEW IF EXISTS `v01inverter_ebox`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8mb4;
/*!50001 CREATE VIEW `v01inverter_ebox` AS SELECT
 1 AS `minute_timestamp`,
  1 AS `original_ts_inverter`,
  1 AS `original_ts_ebox`,
  1 AS `bat_power_scaled`,
  1 AS `bat_soc_scaled`,
  1 AS `output_power_scaled`,
  1 AS `eboxpwr` */;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `weather_data`
--

DROP TABLE IF EXISTS `weather_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `weather_data` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `timestamp` datetime NOT NULL,
  `latitude` decimal(9,6) NOT NULL,
  `longitude` decimal(9,6) NOT NULL,
  `weather` varchar(50) DEFAULT NULL,
  `temperature` decimal(5,2) DEFAULT NULL,
  `created_at` datetime DEFAULT current_timestamp(),
  `humidity` decimal(5,2) DEFAULT NULL,
  `pressure` decimal(7,2) DEFAULT NULL,
  `wind_speed` decimal(5,2) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_timestamp` (`timestamp`)
) ENGINE=InnoDB AUTO_INCREMENT=1394 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Final view structure for view `pv_zaehl00`
--

/*!50001 DROP VIEW IF EXISTS `pv_zaehl00`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_unicode_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`gh`@`192.168.%` SQL SECURITY DEFINER */
/*!50001 VIEW `pv_zaehl00` AS select `pv_zaehl`.`datum` AS `datum`,`pv_zaehl`.`bezug` AS `bezug`,`pv_zaehl`.`einspeis` AS `einspeis` from `pv_zaehl` where `pv_zaehl`.`datum` like '%00:00%' */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `pv_zaehl08`
--

/*!50001 DROP VIEW IF EXISTS `pv_zaehl08`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_unicode_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`gh`@`192.168.%` SQL SECURITY DEFINER */
/*!50001 VIEW `pv_zaehl08` AS select `pv_zaehl`.`datum` AS `datum`,`pv_zaehl`.`bezug` AS `bezug`,`pv_zaehl`.`einspeis` AS `einspeis` from `pv_zaehl` where `pv_zaehl`.`datum` like '%08:00%' */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `pv_zaehl10`
--

/*!50001 DROP VIEW IF EXISTS `pv_zaehl10`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_unicode_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`gh`@`192.168.%` SQL SECURITY DEFINER */
/*!50001 VIEW `pv_zaehl10` AS select `pv_zaehl`.`datum` AS `datum`,`pv_zaehl`.`bezug` AS `bezug`,`pv_zaehl`.`einspeis` AS `einspeis` from `pv_zaehl` where `pv_zaehl`.`datum` like '%10:00%' */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `pv_zaehl19`
--

/*!50001 DROP VIEW IF EXISTS `pv_zaehl19`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_unicode_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`gh`@`192.168.%` SQL SECURITY DEFINER */
/*!50001 VIEW `pv_zaehl19` AS select `pv_zaehl`.`datum` AS `datum`,`pv_zaehl`.`bezug` AS `bezug`,`pv_zaehl`.`einspeis` AS `einspeis` from `pv_zaehl` where `pv_zaehl`.`datum` like '%19:00%' */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `v01inverter_ebox`
--

/*!50001 DROP VIEW IF EXISTS `v01inverter_ebox`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_general_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`gh`@`192.168.%` SQL SECURITY DEFINER */
/*!50001 VIEW `v01inverter_ebox` AS select `i`.`ts_min` AS `minute_timestamp`,`i`.`timestamp` AS `original_ts_inverter`,`e`.`ts` AS `original_ts_ebox`,`i`.`Power_Bat1` * 100 AS `bat_power_scaled`,`i`.`SOC_Bat1` * 10 AS `bat_soc_scaled`,`i`.`ActivePower_Output_Total` * 100 AS `output_power_scaled`,`e`.`Volt` * `e`.`Curr` * 6 / 1000000 AS `eboxpwr` from (`inverter_data` `i` join `pv_ebox2` `e` on(`i`.`ts_min` = `e`.`ts_min`)) */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*M!100616 SET NOTE_VERBOSITY=@OLD_NOTE_VERBOSITY */;

-- Dump completed on 2026-02-17 18:05:01
