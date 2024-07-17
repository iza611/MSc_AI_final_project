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

from LoadImages import LoadImages
from help_functions import hagrid_xywh_to_xyxy, convert_xyxy_to_yolo_xywh, crop_n_transform

# Add the new path to the system path
base_path = abspath('./../training/model/yolov7/')
sys.path.append(base_path)

from models.experimental import attempt_load
# from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, TracedModel
from utils.plots import plot_one_box, plot_one_yolo_box


def detect():
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    
    # Prep save dirs
    save_path = "./../datasets/HaGRID_test_converted/"
    split_set = source.split('/')[-2]   # 'train', 'test', 'val' 

    imgs_save_path = join(save_path, 'images', split_set)
    labels_save_path = join(save_path, 'labels', split_set)
    
    makedirs(imgs_save_path, exist_ok=True)
    makedirs(labels_save_path, exist_ok=True)
        
    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get all gestures bbxs info
    hagrid_annotations = {}
    with open(join(source), 'r') as f:
        data = json.load(f)

        # Check if images in the label file are in the test set
        for image_id, info in data.items():
            hagrid_annotations[image_id] = info
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for path, img, im0s, vid_cap in tqdm(dataset):
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

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} person{'s' * (n > 1)}, "  # add to string
                
                # print(det)
                
                best_bbox = None
                highest_conf = 0

                # Iterate through the detections
                for *xyxy, conf, _ in det:
                    conf = conf.item()  # Convert confidence tensor to float

                    # Update if the current confidence is higher than the highest found
                    if conf > highest_conf:
                        highest_conf = conf
                        best_bbox = xyxy
                
            else:
                best_bbox = torch.tensor([0, 0, im0.shape[1], im0.shape[0]], device='cuda:0')
        
        # Get ground truth gestures bbxs in xyxy format
        gestures_info = hagrid_annotations[p.stem]
        gestures_bbxs = hagrid_xywh_to_xyxy(gestures_info['bboxes'], im0.shape)
        
        # crop around the human
        human_bbx = [int(tensor.item()) for tensor in best_bbox]
        gestures_bbxs = [[int(value) for value in sublist] for sublist in gestures_bbxs]
        cropped_img, transformed_bbxs = crop_n_transform(im0, human_bbx, gestures_bbxs)
        
        # final touch ups and save         
        img_save_path = join(imgs_save_path, p.name)
        label_save_path = join(labels_save_path, p.stem+'.txt')
        
        labels_yolo = convert_xyxy_to_yolo_xywh(transformed_bbxs, gestures_info['labels'], cropped_img.shape)
        
        # SAVE 
        cv2.imwrite(img_save_path, cropped_img)

        with open(label_save_path, 'w') as f:
            for label in labels_yolo:
                f.write(' '.join(map(str, label)) + '\n')
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    detect()
