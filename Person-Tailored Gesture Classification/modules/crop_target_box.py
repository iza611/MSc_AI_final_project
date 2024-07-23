from pathlib import Path
import torch

def crop_target_box(person_bboxs, im0s):
    
    if not len(person_bboxs):
        return -1, [0,0,0,0]
    
    elif len(person_bboxs) == 1:
        x1, y1, x2, y2, _, _ = person_bboxs[0]
    
    else:
        # person_bboxs = sorted(person_bboxs, key=lambda x: x[4], reverse=True)
        # x1, y1, x2, y2, _, _ = person_bboxs[0]
        
        person_bbox = max(person_bboxs, key=lambda x: x[4])
        x1, y1, x2, y2, _, _ = person_bbox
    
    # Ensure valid bounding box coordinates
#     x1, x2 = min(x1, x2), max(x1, x2)
#     y1, y2 = min(y1, y2), max(y1, y2)
    
#     # Handle cases where the bounding box might be out of image bounds
#     x1 = int(max(0, x1))
#     y1 = int(max(0, y1))
#     x2 = int(min(im0s.shape[1], x2))
#     y2 = int(min(im0s.shape[0], y2))
    
#     print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cropped_image = im0s[y1:y2, x1:x2].copy()

    return cropped_image, [x1, y1, x2, y2]