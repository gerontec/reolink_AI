#!/home/gh/python/venv_py311/bin/python3
"""
Minimal CUDA 11 Test für InsightFace mit watchdog2.py Konfiguration
"""
import os
import sys

# KRITISCH: Environment-Variablen ZUERST setzen (wie in watchdog2.py)
os.environ['ORT_DISABLE_CUDNN_FRONTEND'] = '1'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import onnxruntime as ort
from insightface.app import FaceAnalysis

print("=" * 70)
print("CUDA 11 Test - InsightFace mit watchdog2.py Konfiguration")
print("=" * 70)

# 1. ONNX Runtime prüfen
print(f"\n1. ONNX Runtime Version: {ort.__version__}")
print(f"   Available Providers: {ort.get_available_providers()}")

# 2. CUDA-Optionen wie in watchdog2.py
print("\n2. Initialisiere InsightFace mit Tesla P4 CUDA-Optionen...")
cuda_options = {
    'device_id': 0,
    'cudnn_conv_algo_search': 'DEFAULT',
    'gpu_mem_limit': 6 * 1024 * 1024 * 1024,  # 6GB für P4
    'arena_extend_strategy': 'kSameAsRequested',
}

try:
    # Exakt wie in watchdog2.py
    face_app = FaceAnalysis(
        name='buffalo_l',
        providers=[('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
    )

    # Prepare mit det_size=1280 (wie in watchdog2.py)
    face_app.prepare(ctx_id=0, det_size=(1280, 1280))

    # Provider-Check
    providers = face_app.det_model.session.get_providers()

    print(f"\n3. InsightFace Detection Model Providers:")
    print(f"   {providers}")

    # Recognition Model auch prüfen
    if face_app.rec_model:
        rec_providers = face_app.rec_model.session.get_providers()
        print(f"\n4. InsightFace Recognition Model Providers:")
        print(f"   {rec_providers}")

    # Ergebnis
    print("\n" + "=" * 70)
    if 'CUDAExecutionProvider' in providers:
        print("✅ ERFOLG: GPU-Acceleration ist AKTIV")
        print("   InsightFace läuft auf CUDA/GPU")
    else:
        print("❌ FEHLER: GPU-Acceleration ist NICHT aktiv")
        print("   InsightFace läuft nur auf CPU")
        print("\n   Mögliche Ursachen:")
        print("   - CUDA 11 Libraries fehlen (libcublasLt.so.11)")
        print("   - onnxruntime-gpu nicht korrekt installiert")
        print("   - CUDA_PATH nicht gesetzt")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ FEHLER beim Initialisieren von InsightFace:")
    print(f"   {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
