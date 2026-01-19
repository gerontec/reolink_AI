"""
Watchdog Feature Modules
"""
import yaml
import importlib
import os

class ModuleManager:
    def __init__(self, config_path='config.yaml'):
        if not os.path.exists(config_path):
            print(f"⚠️ Config nicht gefunden: {config_path}")
            self.config = {'features': {}}
        else:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        
        self.modules = {}
        self.load_modules()
    
    def load_modules(self):
        """Lädt nur aktivierte Module"""
        features = self.config.get('features', {})
        
        for feature, enabled in features.items():
            if enabled:
                try:
                    module = importlib.import_module(f'modules.{feature}')
                    self.modules[feature] = module.FeatureModule(self.config)
                    print(f"✅ Loaded: {feature}")
                except Exception as e:
                    print(f"❌ Failed to load {feature}: {e}")
    
    def process_detection(self, frame, detection_data):
        """Ruft alle Module auf"""
        results = {}
        for name, module in self.modules.items():
            try:
                result = module.process(frame, detection_data)
                if result:
                    results[name] = result
            except Exception as e:
                print(f"Error in {name}: {e}")
        return results
    
    def get_alarm_rules(self):
        """Sammelt Alarm-Regeln"""
        rules = {}
        for name, module in self.modules.items():
            if hasattr(module, 'get_alarm_rules'):
                rules.update(module.get_alarm_rules())
        return rules
