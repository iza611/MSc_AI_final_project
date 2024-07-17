import glob
import os
import random
from pathlib import Path
import cv2
import numpy as np
import json

import sys
from os.path import abspath, join, dirname
base_path = abspath('./../training/model/yolov7/')
sys.path.append(base_path)

from utils.datasets import letterbox

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

class LoadImages:  
    def __init__(self, path, img_size=640, stride=32):
        
        # path is directing to the json file with labels
        image_ids = []
        with open(path, 'r') as f:
            data = json.load(f)
            for image_id, info in data.items():
                image_ids.append(image_id)

        image_ids_set = set(image_ids)
        print(f'found {len(image_ids)} image IDs in {path}')
        print(f'ended up with {len(image_ids_set)} IDs after converting to a set')
        
        images = []
        for image_id in image_ids_set:
            image_path = join(dirname(dirname(path)), 'like', image_id + '.jpg')
            images.append(image_path)
        
        print(f'list image path length = {len(images)}')
        ni = len(images)

        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = ni  # number of files
        self.video_flag = False
        self.mode = 'image'
        self.cap = None
        assert self.nf > 0, f'No images found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, 'Image Not Found ' + path

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap