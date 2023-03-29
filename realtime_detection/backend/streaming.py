import cv2

from realtime_detection.backend.consts import WEBCAM_URL


def start_video_streaming() -> cv2.VideoCapture:
    """
    Function used to start the video capture from the webcam.
    Returns:
        The video capture object.
    """
    cap = cv2.VideoCapture(WEBCAM_URL)
    return cap
