
import sys
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import cv2
from PIL import Image

print("=" * 45)
print("       SIGNBRIDGE ENVIRONMENT TEST")
print("=" * 45)

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Scikit-learn:", sklearn.__version__)
print("OpenCV:", cv2.__version__)
print("Pillow:", Image.__version__)

print("\nAll initial libraries imported successfully!")
print("SignBridge environment is working.")
