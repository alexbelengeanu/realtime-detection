import inject
import cv2
import numpy as np

from PIL import Image

from realtime_detection.backend.injection import configure_injections
from realtime_detection.backend.streaming import start_video_streaming
from realtime_detection.backend.utils import get_predict
from realtime_detection.models.model import Yolov5Model, MaskRCNNModel
from realtime_detection.models.model_types import ModelTypesEnum


@inject.autoparams("yolov5", "maskrcnn")
def process(yolov5: Yolov5Model, maskrcnn: MaskRCNNModel):
    """
    Process the video stream.
    Args:
        yolov5: The YOLOv5 model.
        maskrcnn: The MaskRCNN model.
    """
    # Open the video stream.
    stream = start_video_streaming()

    # Check if the video stream was opened successfully.
    if not stream.isOpened():
        print("Webcam could not be accessed. Terminated.")
        exit()
    while True:
        # Read the video stream frame by frame.
        ret, source = stream.read()

        # Check if the frame was read successfully.
        if not ret:
            print("A frame could not be read. Terminated.")
            break

        # Create a deep copy of the frame for each model.
        _yolov5_copy = np.copy(source)
        _maskrcnn_copy = np.copy(source)

        # Get the predictions from each model.
        yolov5_prediction = get_predict(model=yolov5,
                                        frame=_yolov5_copy,
                                        model_type=ModelTypesEnum.YOLOV5)

        maskrcnn_prediction = get_predict(model=maskrcnn,
                                          frame=_maskrcnn_copy,
                                          model_type=ModelTypesEnum.MASKRCNN)

        # MaskRCNN was trained on 91 classes from COCO, compared to YOLOv5 which was trained on only the first 80 classes.
        # MaskRCNN prediction is None if the model predicts one of the classes that the YOLOv5 model was not trained on.
        if maskrcnn_prediction is not None:
            horizontal = np.concatenate((yolov5_prediction, maskrcnn_prediction), axis=1)
            cv2.imshow("YOLOv5 [left] | MaskRCNN [right]", horizontal)

        # Press 'q' to exit the infinite loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources.
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    configure_injections()
    process()
