"""
Module to run MegaDetector v5, a PyTorch YOLOv5 (Ultralytics) animal detection model,
on images.
"""

#%% Imports
import sys
import torch
import numpy as np

import ct_utils

CONF_DIGITS = 3
COORD_DIGITS = 4
from md_utils.general import non_max_suppression, scale_coords, xyxy2xywh
from md_utils.augmentations import letterbox
# try:
#     # import pre- and post-processing functions from the YOLOv5 repo https://github.com/ultralytics/yolov5
#     from detection.utils.general import non_max_suppression, scale_coords, xyxy2xywh
#     from detection.utils.augmentations import letterbox
# except ModuleNotFoundError as e:
#     raise ModuleNotFoundError('Could not import YOLOv5 functions.')

#%% Classes

class PTDetector:

    IMAGE_SIZE = 1280  # image size used in training
    STRIDE = 64

    def __init__(self, model_path: str, force_cpu: bool = False):
        gpu_use = torch.cuda.is_available()
        print(f'GPU available: {gpu_use}')
        if gpu_use and not force_cpu:
            self.device = torch.device('cuda:0')
        else:
            self.device = 'cpu'
        self.model = PTDetector._load_model(model_path, self.device)
        if (self.device != 'cpu') and gpu_use:
            self.model.to(self.device)

    @staticmethod
    def _load_model(model_pt_path, device):
        checkpoint = torch.load(model_pt_path, map_location=device)
        model = checkpoint['model'].float().fuse().eval()  # FP32 model

        return model

    def generate_detections_one_image(self, img_original, image_id, detection_threshold, label_map={}):
        """Apply the detector to an image.

        Args:
            img_original: the PIL Image object with EXIF rotation taken into account
            image_id: a path to identify the image; will be in the "file" field of the output object
            detection_threshold: confidence above which to include the detection proposal
            label_map: optional, mapping the numerical label to a string name. The type of the numerical label

        Returns:
            - 'file' (always present)
            - 'max_detection_conf'
            - 'detections', which is a list of detection objects containing keys 'category', 'conf' and 'bbox'
            - 'failure'
        """

        result = {
            "file": image_id
        }
        detections = []

        try:
            img_original = np.asarray(img_original)

            # padded resize
            img = letterbox(img_original, new_shape=PTDetector.IMAGE_SIZE,
                                 stride=PTDetector.STRIDE, auto=True)[0]  # JIT requires auto=False
            img = img.transpose((2, 0, 1))  # HWC to CHW; PIL Image is RGB already
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            img = img.to(self.device)
            img = img.float()
            img /= 255

            if len(img.shape) == 3:  # always true for now, TODO add inference using larger batch size
                img = torch.unsqueeze(img, 0)

            pred: list = self.model(img)[0]

            # NMS
            pred = non_max_suppression(prediction=pred, conf_thres=detection_threshold)

            # format detections/bounding boxes
            gn = torch.tensor(img_original.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for det in pred:
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_original.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        # normalized center-x, center-y, width and height
                        #if int(cls) != 2:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                        api_box = ct_utils.convert_yolo_to_xywh(xywh)

                        conf = ct_utils.truncate_float(conf.tolist(), precision=CONF_DIGITS)

                        # MegaDetector output format's categories start at 1, but this model's start at 0
                        cls = int(cls.tolist()) + 1
                        if cls not in (1, 2, 3):
                            raise KeyError(f'{cls} is not a valid class.')

                        clss = str(cls)
                        label = label_map[clss] if clss in label_map else clss

                        detections.append({
                            "best_class": clss,
                            "best_probability": conf,
                            "name": f'{label} {round(100 * conf)}%',
                            "bbox": ct_utils.truncate_float_array(api_box, precision=COORD_DIGITS)
                        })


        except Exception as e:
            result["failure"] = "Failure inference"

        result["prediction"] = detections

        return result
