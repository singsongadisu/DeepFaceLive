import onnxruntime as ort
import numpy as np

print("Testing ONNX Runtime...")
try:
    print(f"Available providers: {ort.get_available_providers()}")
    # Try to create a session with CPU only
    sess_options = ort.SessionOptions()
    # Dummy session or just check providers
    if 'CPUExecutionProvider' in ort.get_available_providers():
        print("CPUExecutionProvider is available.")
    else:
        print("CPUExecutionProvider is NOT available!")
except Exception as e:
    print(f"Crashed during initialization: {e}")
