import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights


class Yolov5Model:
    def __init__(self,
                 device: torch.device
                 ) -> None:
        """
        Initialize a wrapper over the YOLOv5 model.
        Args:
            device: The device on which the model will be loaded.
        """
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)
        self.model = self.model.eval()


class MaskRCNNModel:
    def __init__(self,
                 device: torch.device
                 ) -> None:
        """
        Initialize a wrapper over the MaskRCNN model.
        Args:
            device: The device on which the model will be loaded.
        """
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.device = device
        self.transforms = weights.transforms()

        self.model = maskrcnn_resnet50_fpn(weights=weights, progress=False).to(device)
        self.model = self.model.eval()
