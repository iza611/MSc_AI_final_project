import torch
import torch.nn as nn
from pathlib import Path

import sys
from os.path import abspath, join, dirname
base_path = abspath('./../training/model/yolov7/')
sys.path.append(base_path)

from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import TracedModel


def load_model(weights_path, device, half, g):
    
    model = attempt_load(weights=weights_path, map_location=device)
    stride = int(model.stride.max())  # model stride
    img_size = 1024
    imgsz = check_img_size(img_size, s=stride) 
    
    # model = TracedModel(model, device, img_size, g)
    
    if half:
        model.half() 
        
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  
        
    return model, stride

def load_targets_model(target, device, half):
    
    weights = f'./trained_detectors/target_{target}.pt'
    model, stride = load_model(weights, device, half, g=False)
    
    return model, stride

def load_gestures_model(device, half):
    
    weights = f'./trained_detectors/gestures.pt'
    model, stride = load_model(weights, device, half, g=True)
    
    return model, stride