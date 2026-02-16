#!/home/gh/python/venv_py311/bin/python3
"""
IR Mode Auto-Detection
Detects if camera is in IR (night vision) mode based on image characteristics
"""
import cv2
import numpy as np

def is_ir_mode(frame):
    """
    Detect if camera is in IR (infrared/night vision) mode
    
    Checks:
    1. Color variance (IR has almost no color)
    2. Histogram spread (IR has different distribution)
    3. Green channel dominance (IR often uses green sensor)
    
    Returns:
        bool: True if IR mode detected
    """
    # Convert to different color spaces
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Method 1: Check color saturation (IR = very low)
    saturation = hsv[:, :, 1]
    avg_saturation = np.mean(saturation)
    
    # Method 2: Check color variance between channels
    b, g, r = cv2.split(frame)
    color_variance = np.std([np.mean(b), np.mean(g), np.mean(r)])
    
    # Method 3: Check if image is nearly grayscale
    # In IR, all channels should be similar
    channel_diff = np.mean(np.abs(r.astype(float) - b.astype(float)))
    
    # Decision criteria
    is_ir = (
        avg_saturation < 15 and      # Very low color saturation
        color_variance < 5 and        # Channels are very similar
        channel_diff < 10             # RGB channels nearly identical
    )
    
    return is_ir

def get_ir_score(frame):
    """
    Get IR mode confidence score (0-100)
    Higher score = more likely IR mode
    
    Useful for debugging and threshold tuning
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    avg_saturation = np.mean(saturation)
    
    b, g, r = cv2.split(frame)
    color_variance = np.std([np.mean(b), np.mean(g), np.mean(r)])
    channel_diff = np.mean(np.abs(r.astype(float) - b.astype(float)))
    
    # Normalize to 0-100 scale
    sat_score = max(0, 100 - (avg_saturation * 5))
    var_score = max(0, 100 - (color_variance * 15))
    diff_score = max(0, 100 - (channel_diff * 8))
    
    # Weighted average
    total_score = (sat_score * 0.5 + var_score * 0.3 + diff_score * 0.2)
    
    return total_score, {
        'saturation': avg_saturation,
        'color_variance': color_variance,
        'channel_diff': channel_diff,
        'sat_score': sat_score,
        'var_score': var_score,
        'diff_score': diff_score
    }

def get_adaptive_confidence(frame):
    """
    Get adaptive confidence values based on IR detection
    
    Returns:
        dict: {
            'tracking': float,
            'detection': float,
            'resolution': int,
            'mode': str,
            'ir_score': float
        }
    """
    is_ir = is_ir_mode(frame)
    ir_score, _ = get_ir_score(frame)
    
    if is_ir:
        # IR mode: Lower confidence
        return {
            'tracking': 0.10,
            'detection': 0.18,
            'resolution': 1024,
            'mode': 'IR',
            'ir_score': ir_score
        }
    else:
        # Day mode: Normal confidence
        return {
            'tracking': 0.30,
            'detection': 0.25,
            'resolution': 1536,
            'mode': 'DAY',
            'ir_score': ir_score
        }

# Test mode
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test with image file
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"❌ Could not load image: {sys.argv[1]}")
            sys.exit(1)
        
        is_ir = is_ir_mode(img)
        score, details = get_ir_score(img)
        config = get_adaptive_confidence(img)
        
        print(f"\n📷 Image Analysis: {sys.argv[1]}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Mode Detected: {config['mode']}")
        print(f"IR Score: {score:.1f}/100")
        print(f"\nMetrics:")
        print(f"  Saturation: {details['saturation']:.1f} (IR < 15)")
        print(f"  Color Variance: {details['color_variance']:.1f} (IR < 5)")
        print(f"  Channel Diff: {details['channel_diff']:.1f} (IR < 10)")
        print(f"\nRecommended Config:")
        print(f"  CONFIDENCE_TRACKING: {config['tracking']}")
        print(f"  CONFIDENCE_DETECTION: {config['detection']}")
        print(f"  AI_RESOLUTION: {config['resolution']}")
        print()
    else:
        # Test with camera
        print("📷 Testing with camera...")
        print("Connect to RTSP or press Ctrl+C to exit")
        print()
        
        cap = cv2.VideoCapture(0)  # Or RTSP URL
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                config = get_adaptive_confidence(frame)
                
                # Display
                text = f"Mode: {config['mode']} | IR Score: {config['ir_score']:.0f}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                cv2.imshow('IR Detection Test', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
