from pathlib import Path
import torch

# from yolov7.utils.general import non_max_suppression, scale_coords
from utils.general import non_max_suppression, scale_coords

def target_person_detection(model, path, img, im0s, conf_thres):
    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, 0.45, classes=None, agnostic=False)
    # print(pred)

    # Process detections
    for i, det in enumerate(pred):
        # print(det)
        if len(det):
            # Rescale boxes from img_size to im0s size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            det = [
                [item.item() if isinstance(item, torch.Tensor) else item for item in detection]
                for detection in det
            ]

    return det


