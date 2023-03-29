import numpy as np
import cv2

# Webcam connection URL
WEBCAM_URL = "http://86.127.212.219:80/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER"

# Load the COCO class labels our models were trained on
LABELS = open("coco_names.txt").read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Bounding box related constants
RECTANGLE_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.25
FONT_THICKNESS = 3
FONT_LINE_TYPE = cv2.LINE_AA
