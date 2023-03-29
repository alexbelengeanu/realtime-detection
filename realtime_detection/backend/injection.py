import inject
import torch

from realtime_detection.models.model import Yolov5Model, MaskRCNNModel


def configure_yolov5_model(binder: inject.Binder,
                           device: torch.device) -> None:
    """
    Configure the YOLOv5 model singleton object.
    Args:
        binder: The binder object.
        device: The device on which the model will be loaded.
    """
    binder.bind_to_constructor(Yolov5Model, lambda: Yolov5Model(device=device))


def configure_maskrcnn_model(binder: inject.Binder,
                             device: torch.device) -> None:
    """
    Configure the MaskRCNN model singleton object.
    Args:
        binder: The binder object.
        device: The device on which the model will be loaded.
    """
    binder.bind_to_constructor(MaskRCNNModel, lambda: MaskRCNNModel(device=device))


def config_singletons(binder: inject.Binder) -> None:
    """
    Configure all the singleton objects.
    Args:
        binder: The binder object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configure_yolov5_model(binder, device)
    configure_maskrcnn_model(binder, device)


def configure_injections() -> None:
    """
    Configure all the injected variables.
    """
    inject.configure(config_singletons)
