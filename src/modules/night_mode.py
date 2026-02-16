"""
Night Mode Module
"""
from datetime import datetime

class FeatureModule:
    def __init__(self, config):
        self.config = config.get('night_mode', {})
        self.start_hour = self.config.get('start_hour', 22)
        self.end_hour = self.config.get('end_hour', 6)
        self.sensitivity_boost = self.config.get('sensitivity_boost', 0.1)
        self.alarm_all = self.config.get('alarm_all_motion', True)
        print(f"  Night: {self.start_hour}:00-{self.end_hour}:00")
    
    def process(self, frame, detection_data):
        return {'is_night': self.is_night_time()}
    
    def is_night_time(self):
        hour = datetime.now().hour
        if self.start_hour > self.end_hour:
            return hour >= self.start_hour or hour < self.end_hour
        else:
            return self.start_hour <= hour < self.end_hour
    
    def get_alarm_rules(self):
        if self.is_night_time():
            return {
                'is_night': True,
                'confidence_boost': self.sensitivity_boost,
                'alarm_all_motion': self.alarm_all,
                'priority': 'high'
            }
        return {'is_night': False}
