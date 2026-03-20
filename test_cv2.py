import cv2
import numpy as np
import sys

print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

try:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_resized = cv2.resize(img, (50, 50))
    print("cv2.resize successful")
    
    # Test np.int/np.float just in case I missed any
    try:
        print(f"np.int: {np.int}")
    except AttributeError:
        print("np.int is missing (expected in NumPy 2.0)")
        
    try:
        print(f"np.float: {np.float}")
    except AttributeError:
        print("np.float is missing (expected in NumPy 2.0)")

    print("Diagnostic complete.")
except Exception as e:
    print(f"Diagnostic failed: {e}")
