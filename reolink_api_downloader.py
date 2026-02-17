#!/home/gh/python/venv_py311/bin/python3
"""
Reolink E1 Pro SD Card Video Downloader (using reolink-aio library)
Downloads motion-triggered recordings directly from camera SD card via local API
NO cloud account needed - works 100% locally!
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict

try:
    from reolink_aio.api import Host
except ImportError:
    print("❌ ERROR: reolink-aio library not installed!")
    print("Install with: pip install reolink-aio")
    sys.exit(1)

# Configuration
CAMERA_IP = "192.168.178.128"
CAMERA_PORT = 8000  # HTTP API port (found via nmap)
USERNAME = "web1"  # Camera username
PASSWORD = "Auchgut11"  # Camera password

OUTPUT_BASE = Path("/var/www/web1")
CAMERA_NAME = "Cam2"
STATE_FILE = Path("/home/gh/python/reolink_last_download.json")

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('/home/gh/python/logs/reolink_api.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


def load_last_download_time() -> datetime:
    """Load timestamp of last successful download"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                timestamp = data.get('last_download')
                if timestamp:
                    return datetime.fromisoformat(timestamp)
        except Exception as e:
            logger.warning(f"Could not load state file: {e}")

    # Default: last 24 hours
    return datetime.now() - timedelta(hours=24)


def save_last_download_time(timestamp: datetime):
    """Save timestamp of last successful download"""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump({
                'last_download': timestamp.isoformat(),
                'camera': CAMERA_NAME
            }, f)
    except Exception as e:
        logger.error(f"Could not save state file: {e}")


def get_output_path(filename: str, start_time: datetime) -> Path:
    """Generate output path matching FTP structure: /var/www/web1/YYYY/MM/Cam2_*.mp4"""
    year = start_time.year
    month = f"{start_time.month:02d}"

    # Generate new filename
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    new_filename = f"{CAMERA_NAME}_01_{timestamp}.mp4"

    return OUTPUT_BASE / str(year) / month / new_filename


async def download_recordings(host: Host, start_time: datetime, end_time: datetime) -> int:
    """Download all recordings from SD card within time range"""
    downloaded_count = 0
    latest_time = start_time

    try:
        logger.info(f"📅 Searching recordings from {start_time} to {end_time}")

        # Search for recordings on SD card
        # reolink-aio uses different method - we'll search day by day
        files = await host.get_vod_source(start_time, end_time)

        if not files:
            logger.info("📭 No recordings found on SD card")
            return 0

        logger.info(f"📹 Found {len(files)} recordings")

        # Download each file
        for file_info in files:
            try:
                # Extract file information
                file_name = file_info.get('name', '')
                file_start = file_info.get('start_time')
                file_size = file_info.get('size', 0)

                if not file_name or not file_start:
                    logger.warning(f"⚠️  Skipping invalid file entry: {file_info}")
                    continue

                # Generate output path
                output_path = get_output_path(file_name, file_start)

                # Skip if already exists
                if output_path.exists():
                    logger.info(f"⏭️  Skipping existing: {output_path.name}")
                    continue

                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"⬇️  Downloading: {file_name} ({file_size} bytes)")

                # Download file from SD card
                content = await host.get_vod(file_name)

                if content:
                    with open(output_path, 'wb') as f:
                        f.write(content)

                    logger.info(f"✅ Downloaded: {output_path.name} ({len(content)} bytes)")
                    downloaded_count += 1

                    # Track latest recording time
                    if file_start > latest_time:
                        latest_time = file_start
                else:
                    logger.error(f"❌ Failed to download: {file_name}")

            except Exception as e:
                logger.error(f"❌ Error downloading {file_info.get('name', 'unknown')}: {e}")
                continue

        # Save state
        if downloaded_count > 0:
            save_last_download_time(latest_time)
            logger.info(f"✅ Downloaded {downloaded_count} new videos")
        else:
            logger.info("ℹ️  All recordings already downloaded")

    except Exception as e:
        logger.error(f"❌ Error searching/downloading recordings: {e}")

    return downloaded_count


async def main():
    """Main download loop"""
    logger.info("=" * 60)
    logger.info(f"🚀 Starting Reolink SD Card Downloader for {CAMERA_NAME}")
    logger.info("=" * 60)

    # Initialize Reolink API client
    host = Host(
        CAMERA_IP,
        USERNAME,
        PASSWORD,
        port=CAMERA_PORT,
        use_https=False  # Use HTTP on port 8000
    )

    try:
        # Login to camera
        logger.info(f"🔐 Connecting to camera at {CAMERA_IP}:{CAMERA_PORT}")

        if not await host.login():
            logger.error("❌ Failed to login to camera")
            logger.error("Check IP, port, username, and password!")
            sys.exit(1)

        logger.info(f"✅ Successfully logged in to {host.api.camera_name()}")

        # Get camera info
        await host.get_states()
        logger.info(f"📷 Camera: {host.api.model}")
        logger.info(f"💾 SD Card: {host.api.hdd_info}")

        # Get time range to search
        last_download = load_last_download_time()
        current_time = datetime.now()

        # Download recordings
        downloaded_count = await download_recordings(host, last_download, current_time)

        logger.info("=" * 60)
        logger.info(f"✨ Finished! Total new videos: {downloaded_count}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Logout
        await host.logout()
        logger.info("🔓 Logged out")


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
