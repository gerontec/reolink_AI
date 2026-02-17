#!/usr/bin/env python3
"""
ONVIF Recording Test Script for Reolink E1 Pro
Tests ONVIF connectivity and searches for SD card recordings
"""

import sys
from datetime import datetime, timedelta
from onvif import ONVIFCamera
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CAMERA_IP = "192.168.178.128"
ONVIF_PORT = 8000  # Standard ONVIF port (NOT 554 which is RTSP)
USERNAME = "web1"
PASSWORD = "Auchgut11"

def test_onvif_connection():
    """Test ONVIF connection and explore available services"""
    logger.info("=" * 60)
    logger.info("🔍 Testing ONVIF Connection to Reolink E1 Pro")
    logger.info("=" * 60)

    try:
        # Create ONVIF camera client
        logger.info(f"📡 Connecting to {CAMERA_IP}:{ONVIF_PORT}")

        # Find WSDL directory from onvif-zeep package
        import os
        try:
            import onvif
            onvif_path = os.path.dirname(onvif.__file__)
            wsdl_dir = os.path.join(onvif_path, 'wsdl')

            if os.path.exists(wsdl_dir):
                logger.info(f"✅ Found WSDL at: {wsdl_dir}")
                mycam = ONVIFCamera(
                    CAMERA_IP,
                    ONVIF_PORT,
                    USERNAME,
                    PASSWORD,
                    wsdl_dir
                )
            else:
                logger.info("⚠️  WSDL not found, using defaults")
                mycam = ONVIFCamera(
                    CAMERA_IP,
                    ONVIF_PORT,
                    USERNAME,
                    PASSWORD
                )
        except Exception as e:
            logger.warning(f"WSDL detection failed: {e}")
            logger.info("Trying without WSDL path...")
            mycam = ONVIFCamera(
                CAMERA_IP,
                ONVIF_PORT,
                USERNAME,
                PASSWORD
            )

        # Get device information
        logger.info("✅ ONVIF connection successful!")

        # Get device info
        device_info = mycam.devicemgmt.GetDeviceInformation()
        logger.info(f"📷 Device: {device_info.Manufacturer} {device_info.Model}")
        logger.info(f"🔧 Firmware: {device_info.FirmwareVersion}")
        logger.info(f"🆔 Serial: {device_info.SerialNumber}")

        # Get available services
        logger.info("\n" + "=" * 60)
        logger.info("📋 Available ONVIF Services:")
        logger.info("=" * 60)

        services = mycam.devicemgmt.GetServices(False)
        for service in services:
            logger.info(f"  • {service.Namespace}")
            logger.info(f"    URL: {service.XAddr}")

        return mycam

    except Exception as e:
        logger.error(f"❌ ONVIF connection failed: {e}")
        logger.error(f"💡 Try these alternative ports: 80, 8999, 8080")
        return None


def search_recordings(mycam):
    """Search for recordings on the camera"""
    logger.info("\n" + "=" * 60)
    logger.info("🎥 Searching for Recordings...")
    logger.info("=" * 60)

    try:
        # Try to get media service
        media_service = mycam.create_media_service()
        logger.info("✅ Media service available")

        # Get profiles
        profiles = media_service.GetProfiles()
        logger.info(f"📊 Found {len(profiles)} media profile(s)")

        for idx, profile in enumerate(profiles):
            logger.info(f"\nProfile {idx + 1}:")
            logger.info(f"  Name: {profile.Name}")
            logger.info(f"  Token: {profile.token}")

    except Exception as e:
        logger.warning(f"⚠️  Media service error: {e}")

    try:
        # Try recording search (ONVIF Profile G)
        logger.info("\n🔍 Attempting recording search (Profile G)...")

        # Create search service
        search_service = mycam.create_search_service()

        # Define search scope (last 24 hours)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)

        # Search for recordings
        search_result = search_service.FindRecordings(
            Scope={
                'IncludedSources': [],
                'IncludedRecordings': [],
                'RecordingInformationFilter': None
            },
            MaxMatches=100,
            KeepAliveTime=timedelta(seconds=30)
        )

        if search_result and hasattr(search_result, 'SearchToken'):
            logger.info(f"✅ Search initiated - Token: {search_result.SearchToken}")

            # Get recording results
            results = search_service.GetRecordingSearchResults(
                SearchToken=search_result.SearchToken,
                MinResults=1,
                MaxResults=100,
                WaitTime=timedelta(seconds=5)
            )

            if results and hasattr(results, 'ResultList'):
                logger.info(f"\n📼 Found {len(results.ResultList.RecordingInformation)} recording(s):")

                for rec in results.ResultList.RecordingInformation:
                    logger.info(f"\n  Recording: {rec.RecordingToken}")
                    logger.info(f"    Start: {rec.EarliestRecording}")
                    logger.info(f"    End: {rec.LatestRecording}")
            else:
                logger.info("ℹ️  No recordings found in last 24 hours")
        else:
            logger.warning("⚠️  Recording search not supported")

    except AttributeError as e:
        logger.warning(f"⚠️  Recording search not available: {e}")
        logger.info("💡 Camera may not support ONVIF Profile G (Recording Search)")

    except Exception as e:
        logger.error(f"❌ Recording search failed: {e}")


def test_rtsp_stream(mycam):
    """Get RTSP stream URL"""
    logger.info("\n" + "=" * 60)
    logger.info("📹 Getting RTSP Stream URL...")
    logger.info("=" * 60)

    try:
        media_service = mycam.create_media_service()
        profiles = media_service.GetProfiles()

        if profiles:
            token = profiles[0].token
            stream_uri = media_service.GetStreamUri({
                'StreamSetup': {
                    'Stream': 'RTP-Unicast',
                    'Transport': {'Protocol': 'RTSP'}
                },
                'ProfileToken': token
            })

            logger.info(f"✅ RTSP Stream URL:")
            logger.info(f"   {stream_uri.Uri}")
            logger.info(f"\n💡 Use VLC or ffmpeg to view/record:")
            logger.info(f"   vlc {stream_uri.Uri}")

    except Exception as e:
        logger.error(f"❌ Failed to get stream URL: {e}")


def main():
    """Main test function"""

    # Test ONVIF connection
    mycam = test_onvif_connection()

    if not mycam:
        logger.error("\n❌ Cannot proceed without ONVIF connection")
        logger.info("\n💡 Troubleshooting:")
        logger.info("   1. Enable ONVIF in Reolink app")
        logger.info("   2. Try port 80, 8999, or 8080")
        logger.info("   3. Check username/password")
        return 1

    # Search for recordings
    search_recordings(mycam)

    # Get RTSP stream
    test_rtsp_stream(mycam)

    logger.info("\n" + "=" * 60)
    logger.info("✅ ONVIF Test Complete")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
