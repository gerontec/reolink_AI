"""
Notification Module
"""
import requests
import io
from PIL import Image
import cv2

class FeatureModule:
    def __init__(self, config):
        self.config = config.get('notifications', {})
        self.telegram = self.config.get('telegram', {})
        
        if self.telegram.get('enabled'):
            self.bot_token = self.telegram.get('bot_token')
            self.chat_id = self.telegram.get('chat_id')
            print(f"  Telegram: {self.chat_id}")
    
    def process(self, frame, detection_data):
        return None
    
    def send_alert(self, message, priority='medium', frame=None):
        """Sendet Telegram Nachricht"""
        if not self.telegram.get('enabled'):
            return False
        
        emoji = {'low': 'ℹ️', 'medium': '⚠️', 'high': '🚨'}
        full_message = f"{emoji.get(priority, '•')} {message}"
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {'chat_id': self.chat_id, 'text': full_message}
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            
            if frame is not None and response.ok:
                self.send_image(frame)
            
            return response.ok
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
    
    def send_image(self, frame):
        """Sendet Snapshot"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        
        # CV2 → JPEG bytes
        _, buffer = cv2.imencode('.jpg', frame)
        
        files = {'photo': ('snapshot.jpg', buffer.tobytes(), 'image/jpeg')}
        data = {'chat_id': self.chat_id}
        
        try:
            requests.post(url, files=files, data=data, timeout=10)
        except Exception as e:
            print(f"Image error: {e}")
