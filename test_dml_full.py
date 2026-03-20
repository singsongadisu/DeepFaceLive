import onnxruntime as ort
import numpy as np
import cv2
import os

print("Testing ONNX Runtime DirectML...")
try:
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    if 'DmlExecutionProvider' not in providers:
        print("DmlExecutionProvider NOT FOUND!")
    
    # Path to S3FD model
    model_path = r'c:\Users\sings\OneDrive\Desktop\Projects\DeepFaceLive\modelhub\onnx\S3FD\S3FD.onnx'
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        # Try to find it
        for root, dirs, files in os.walk(r'c:\Users\sings\OneDrive\Desktop\Projects\DeepFaceLive\modelhub\onnx'):
            if 'S3FD.onnx' in files:
                model_path = os.path.join(root, 'S3FD.onnx')
                print(f"Found model at: {model_path}")
                break

    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        sess = ort.InferenceSession(model_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        print(f"Session created with providers: {sess.get_providers()}")
        
        # Dummy input
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        print(f"Input name: {input_name}, shape: {input_shape}")
        
        # Create dummy image [1, 3, 640, 640]
        dummy_input = np.random.random((1, 3, 640, 640)).astype(np.float32)
        
        print("Running inference...")
        outputs = sess.run(None, {input_name: dummy_input})
        print("Inference successful!")
    else:
        print("S3FD.onnx not found, skipping inference test.")

except Exception as e:
    print(f"Crashed: {e}")
    import traceback
    traceback.print_exc()
