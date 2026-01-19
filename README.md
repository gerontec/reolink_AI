# Reolink AI Watchdog System

Advanced AI-powered surveillance system for Reolink cameras with object detection, face recognition, and intelligent recording.

## Features

- 🎯 **YOLO Object Detection** - Person, car, truck, motorcycle, bus detection
- 👤 **Face Recognition** - Automatic clustering and recognition with GPU acceleration
- 🌙 **Auto IR Detection** - Automatically adjusts for day/night modes
- 📹 **Smart Recording** - Pre/post-recording with configurable triggers
- 📊 **Live Dashboard** - Web-based monitoring with statistics
- 🗄️ **Database Logging** - Complete tracking of all detections
- 🅿️ **Parking Detection** - Identifies parked vehicles
- 🎨 **Video Overlays** - Detection lines and bounding boxes

## Requirements

- Python 3.11+
- CUDA-capable GPU (for face recognition)
- MariaDB/MySQL database
- Reolink camera with RTSP support
- Ubuntu 24.04 (recommended)

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd reolink_AI

# Run installer
bash scripts/install.sh

# Edit configuration
nano config/config.yaml

# Start watchdog
python src/watchdog.py
```

## Configuration

Edit `config/config.yaml` with your settings:

- Camera RTSP URL
- Detection zones
- Recording parameters
- Database credentials
- Face recognition settings

## Usage

### Start Watchdog
```bash
python src/watchdog.py
```

### View Dashboard
```bash
python scripts/dashboard.py
# Open http://your-server/reports/
```

### Manage Face Clusters
```bash
python scripts/face_clusters.py list
python scripts/face_clusters.py rename Unknown_1 "John"
```

### Check Status
```bash
bash scripts/watchdog_status.sh
```

## Architecture

```
reolink_AI/
├── src/
│   ├── watchdog.py          # Main application
│   ├── db_logger.py          # Database interface
│   ├── video_writer.py       # Video recording
│   ├── face_handler.py       # Face recognition
│   └── ir_detector.py        # IR mode detection
├── config/
│   ├── config.yaml           # Configuration
│   └── schema.sql            # Database schema
├── scripts/
│   ├── dashboard.py          # Web dashboard
│   ├── cleanup_logs.sh       # Log maintenance
│   └── install.sh            # Installation
└── docs/                     # Documentation
```

## Database Schema

See `config/schema.sql` for complete database structure.

Main tables:
- `cam_video_archive` - Recorded videos
- `cam_detections` - Object detections
- `cam_face_recognitions` - Face recognition events
- `cam_face_embeddings` - Face embeddings database

## Performance

- Detection: ~100ms per frame (4K input)
- Face Recognition: ~50ms per person
- Recording: H.264 @ 15 FPS, browser-compatible
- Storage: ~1-2 MB/minute video

## Troubleshooting

### No person detection at night
- Check IR mode auto-detection: `python src/ir_detector.py`
- Lower confidence thresholds in config

### Videos not saved
- Check `BASE_RECORD_DIR` path
- Verify directory permissions: `chmod 2775 /var/www/aufnahmen`

### Face recognition not working
- Verify GPU: `nvidia-smi`
- Check numpy version: `pip list | grep numpy` (must be <2.0)

## Contributing

Pull requests welcome! Please ensure:
- Code follows existing style
- All tests pass
- Documentation updated

## License

MIT License - See LICENSE file

## Credits

Built with:
- Ultralytics YOLOv8
- facenet-pytorch
- OpenCV
- PyTorch

---
Made with ❤️ for home security
