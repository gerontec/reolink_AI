#!/home/gh/python/venv_py311/bin/python3
"""
Reolink E1 Pro SD Card Video Downloader
Downloads motion-triggered recordings directly from camera SD card via local API
NO cloud account needed - works 100% locally!
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import urllib3

# Disable SSL warnings for self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
CAMERA_IP = "192.168.178.128"
CAMERA_PORT = 8000  # HTTP API port (found via nmap)
USERNAME = "admin"  # Change to your camera username
PASSWORD = "your_password_here"  # Change to your camera password

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


class ReolinkCamera:
    """Reolink Camera API Client (Local, No Cloud)"""

    def __init__(self, ip: str, username: str, password: str, port: int = 80):
        self.ip = ip
        self.username = username
        self.password = password
        self.port = port
        self.token = None
        self.base_url = f"http://{ip}:{port}"

    def login(self) -> bool:
        """Login to camera and get token"""
        try:
            url = f"{self.base_url}/api.cgi?cmd=Login"
            payload = [{
                "cmd": "Login",
                "action": 0,
                "param": {
                    "User": {
                        "userName": self.username,
                        "password": self.password
                    }
                }
            }]

            response = requests.post(url, json=payload, timeout=10, verify=False)
            response.raise_for_status()

            data = response.json()
            if data and len(data) > 0:
                result = data[0]
                if result.get("code") == 0:
                    self.token = result["value"]["Token"]["name"]
                    logger.info(f"✅ Login successful to {self.ip}")
                    return True
                else:
                    logger.error(f"❌ Login failed: {result.get('error', 'Unknown error')}")
                    return False

            logger.error("❌ Invalid login response")
            return False

        except Exception as e:
            logger.error(f"❌ Login error: {e}")
            return False

    def search_recordings(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Search for recordings on SD card"""
        if not self.token:
            logger.error("Not logged in")
            return []

        try:
            url = f"{self.base_url}/api.cgi?cmd=Search&token={self.token}"
            payload = [{
                "cmd": "Search",
                "action": 0,
                "param": {
                    "Search": {
                        "channel": 0,  # Channel 0 for E1 Pro
                        "onlyStatus": 0,
                        "streamType": "main",  # "main" for best quality
                        "StartTime": {
                            "year": start_time.year,
                            "mon": start_time.month,
                            "day": start_time.day,
                            "hour": start_time.hour,
                            "min": start_time.minute,
                            "sec": start_time.second
                        },
                        "EndTime": {
                            "year": end_time.year,
                            "mon": end_time.month,
                            "day": end_time.day,
                            "hour": end_time.hour,
                            "min": end_time.minute,
                            "sec": end_time.second
                        }
                    }
                }
            }]

            response = requests.post(url, json=payload, timeout=30, verify=False)
            response.raise_for_status()

            data = response.json()
            if data and len(data) > 0:
                result = data[0]
                if result.get("code") == 0:
                    files = result["value"]["SearchResult"]["File"]
                    logger.info(f"📹 Found {len(files)} recordings")
                    return files
                else:
                    logger.warning(f"Search returned code {result.get('code')}")
                    return []

            return []

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def download_file(self, file_info: Dict, output_path: Path) -> bool:
        """Download a recording file from SD card"""
        if not self.token:
            logger.error("Not logged in")
            return False

        try:
            # Parse file time
            start_time = file_info["StartTime"]
            filename = file_info["name"]

            # Build download URL
            url = f"{self.base_url}/cgi-bin/api.cgi"
            params = {
                "cmd": "Download",
                "token": self.token,
                "source": filename,
                "output": filename
            }

            logger.info(f"⬇️  Downloading: {filename} ({file_info['size']} bytes)")

            response = requests.get(url, params=params, stream=True, timeout=300, verify=False)
            response.raise_for_status()

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress
            downloaded = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            logger.info(f"✅ Downloaded: {output_path.name} ({downloaded} bytes)")
            return True

        except Exception as e:
            logger.error(f"❌ Download error for {file_info.get('name', 'unknown')}: {e}")
            return False

    def logout(self):
        """Logout from camera"""
        if not self.token:
            return

        try:
            url = f"{self.base_url}/api.cgi?cmd=Logout&token={self.token}"
            requests.post(url, timeout=5, verify=False)
            logger.info("🔓 Logged out")
        except:
            pass


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


def get_output_path(file_info: Dict) -> Path:
    """Generate output path matching FTP structure: /var/www/web1/YYYY/MM/Cam2_*.mp4"""
    start_time = file_info["StartTime"]
    year = start_time["year"]
    month = f"{start_time['mon']:02d}"

    # Parse original filename to get timestamp
    # Example: "20260217_123045.mp4" -> "Cam2_01_20260217_123045.mp4"
    original_name = file_info["name"]

    # Generate new filename
    timestamp = f"{year}{month:02d}{start_time['day']:02d}_{start_time['hour']:02d}{start_time['min']:02d}{start_time['sec']:02d}"
    new_filename = f"{CAMERA_NAME}_01_{timestamp}.mp4"

    return OUTPUT_BASE / str(year) / month / new_filename


def main():
    """Main download loop"""
    logger.info("=" * 60)
    logger.info(f"🚀 Starting Reolink SD Card Downloader for {CAMERA_NAME}")
    logger.info("=" * 60)

    # Validate configuration
    if PASSWORD == "your_password_here":
        logger.error("❌ Please configure USERNAME and PASSWORD in the script!")
        sys.exit(1)

    # Initialize camera client
    camera = ReolinkCamera(CAMERA_IP, USERNAME, PASSWORD, CAMERA_PORT)

    # Login
    if not camera.login():
        logger.error("❌ Failed to login to camera")
        sys.exit(1)

    try:
        # Get time range to search
        last_download = load_last_download_time()
        current_time = datetime.now()

        logger.info(f"📅 Searching recordings from {last_download} to {current_time}")

        # Search for recordings
        recordings = camera.search_recordings(last_download, current_time)

        if not recordings:
            logger.info("📭 No new recordings found")
            return

        # Download each recording
        downloaded_count = 0
        latest_time = last_download

        for file_info in recordings:
            output_path = get_output_path(file_info)

            # Skip if already exists
            if output_path.exists():
                logger.info(f"⏭️  Skipping existing: {output_path.name}")
                continue

            # Download
            if camera.download_file(file_info, output_path):
                downloaded_count += 1

                # Track latest recording time
                file_time = datetime(
                    file_info["StartTime"]["year"],
                    file_info["StartTime"]["mon"],
                    file_info["StartTime"]["day"],
                    file_info["StartTime"]["hour"],
                    file_info["StartTime"]["min"],
                    file_info["StartTime"]["sec"]
                )
                if file_time > latest_time:
                    latest_time = file_time

        # Save state
        if downloaded_count > 0:
            save_last_download_time(latest_time)
            logger.info(f"✅ Downloaded {downloaded_count} new videos")
        else:
            logger.info("ℹ️  All recordings already downloaded")

    finally:
        camera.logout()

    logger.info("=" * 60)
    logger.info(f"✨ Finished! Total new videos: {downloaded_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
