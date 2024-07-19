import sys
from os.path import abspath, exists, join
from os import makedirs
base_path = abspath('./../training/model/yolov7/')
sys.path.append(base_path)

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, TracedModel
from utils.plots import plot_one_box, plot_one_yolo_box

from pathlib import Path
import torch
# import torch.backends.cudnn as cudnn
from numpy import random
import sys
from os.path import abspath, exists, join
from os import makedirs
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import argparse
import json

def gesture_recognition(person_extracted_imgs):
    device = select_device('0')
    half = device.type != 'cpu' 

    model = attempt_load(weights=f'./trained_detectors/gestures.pt', map_location=device)
    stride = int(model.stride.max()) 
    imgsz = check_img_size(imgsz, s=stride) 
    
    model = TracedModel(model, device, opt.img_size)
    if half:
        model.half() 
        
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  
        
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    gestures_bboxs = {}
    
    for img_name, img in tqdm(person_extracted_imgs.items()):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path        p.name=img.jpg
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                gestures_bboxs[p.name] = det
            else:
                gestures_bboxs[p.name] = []
    
    return gestures_bboxs