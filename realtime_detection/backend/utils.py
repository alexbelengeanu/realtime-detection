import cv2
import numpy as np
import torch

from typing import List, Dict, Union
from PIL import Image

from realtime_detection.backend.consts import COLORS, LABELS, FONT, FONT_SCALE, FONT_THICKNESS, FONT_LINE_TYPE,\
    RECTANGLE_THICKNESS
from realtime_detection.models.model import Yolov5Model, MaskRCNNModel
from realtime_detection.models.model_types import ModelTypesEnum


def draw_maskrcnn_bboxes(source: np.ndarray, result: List[Dict[str, torch.Tensor]]) -> np.ndarray:
    """
    Draw bounding boxes based on the MaskRCNN prediction.
    Args:
        source: The source image.
        result: The MaskRCNN prediction list.
    Returns:
        The image with the bounding boxes drawn.
    """
    for index, score in enumerate(result[0]['scores']):
        # Get bounding box coordinates
        bbox_coordinates = result[0]['boxes'][index]
        x_min, y_min, x_max, y_max = int(bbox_coordinates[0]), int(bbox_coordinates[1]), \
            int(bbox_coordinates[2]), int(bbox_coordinates[3])

        start_point = (x_min, y_min)
        end_point = (x_max, y_max)

        # Get bounding box color
        label = int(result[0]['labels'][index])

        # The MaskRCNN model was trained on 91 object categories from COCO dataset.
        # The YOLOv5 model was trained on only the first 80 classes from COCO.
        # We will skip all classes that are not in the YOLOv5 training dataset.
        if label < 80:
            blue = int(COLORS[label][0])
            green = int(COLORS[label][1])
            red = int(COLORS[label][2])
            color = (blue, green, red)

            # Get text information
            label = LABELS[int(result[0]['labels'][index]) - 1]
            text_position = (x_min, y_min - 5)

            # Draw only bounding boxes with a confidence greater than 0.5
            if score > .5:
                source = cv2.rectangle(source,
                                       start_point,
                                       end_point,
                                       color,
                                       RECTANGLE_THICKNESS)
                source = cv2.putText(source,
                                     label,
                                     text_position,
                                     FONT,
                                     FONT_SCALE,
                                     color,
                                     FONT_THICKNESS,
                                     FONT_LINE_TYPE)

        else:
            return None

    return resize(source, 640, 480)


def resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize the frame to the specified width and height so that it can be displayed nicely in the window.
    Args:
        frame: The frame to be resized.
        width: The width of the resized frame.
        height: The height of the resized frame.
    Returns:
        The resized frame.
    """
    resized = cv2.resize(frame, (width, height))
    return resized


def get_yolov5_predict(model: Yolov5Model, frame: np.ndarray) -> np.ndarray:
    """
    Get the YOLOv5 prediction with bounding boxes drawn.
    Args:
        model: The YOLOv5 model.
        frame: The frame to be predicted on.
    Returns:
        The frame with the bounding boxes drawn.
    """
    predict = model.model(frame)
    return resize(predict.render()[0], 640, 480)


def get_maskrcnn_predict(model: MaskRCNNModel, frame: np.ndarray) -> np.ndarray:
    """
    Get the MaskRCNN prediction with bounding boxes drawn.
    Args:
        model: The MaskRCNN model.
        frame: The frame to be predicted on.
    Returns:
        The frame with the bounding boxes drawn.
    """
    frame_copy = Image.fromarray(frame)
    frame_to_tensor = model.transforms(frame_copy)
    frame_to_tensor = frame_to_tensor.unsqueeze(0)
    frame_to_tensor = frame_to_tensor.to(model.device)
    result = model.model(frame_to_tensor)
    predict = draw_maskrcnn_bboxes(source=frame, result=result)
    return predict


def get_predict(model: Union[Yolov5Model, MaskRCNNModel],
                frame: np.ndarray,
                model_type: ModelTypesEnum) -> np.ndarray:
    """
    Get the prediction with bounding boxes drawn.
    Args:
        model: The model.
        frame: The frame to be predicted on.
        model_type: The model type.
    Returns:
        The frame with the bounding boxes drawn.
    """
    if model_type == ModelTypesEnum.YOLOV5:
        return get_yolov5_predict(model, frame)
    elif model_type == ModelTypesEnum.MASKRCNN:
        return get_maskrcnn_predict(model, frame)
    else:
        raise ValueError('Invalid model type.')
