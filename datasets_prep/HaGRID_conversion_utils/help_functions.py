def hagrid_xywh_to_xyxy(bboxes, img_shape):
    """
    Convert bounding boxes from Hagrid format (xywh) to (xyxy).

    Args:
        bboxes (list of list): List of bounding boxes in the format
                               [[top left X, top left Y, width, height], ...].
        img_shape (tuple): Shape of the image as (height, width, channels).

    Returns:
        list of list: List of converted bounding boxes in the format
                       [[x_min, y_min, x_max, y_max], ...].
    """
    
    height, width, _ = img_shape
    xyxy_bboxes = []
    
    for bbox in bboxes:
        x, y, w, h = bbox
        
        # Convert normalized values back to pixel values
        x_min = x * width
        y_min = y * height
        x_max = (x + w) * width
        y_max = (y + h) * height
        
        xyxy_bboxes.append([x_min, y_min, x_max, y_max])
    
    return xyxy_bboxes

def convert_xyxy_to_yolo_xywh(transformed_bboxes, labels_info, img_shape):
    
    label_mapping = {
    'no_gesture': 0,
    'like': 1,
    'ok': 2,
    'palm': 3,
    'two_up': 4
    }
    class_indices = [label_mapping[label.lower()] for label in labels_info]
    img_height = img_shape[0]
    img_width = img_shape[1]
    
    yolo_annotations = []
    for bbox, class_idx in zip(transformed_bboxes, class_indices):
        x_min, y_min, x_max, y_max = bbox
        
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        yolo_annotations.append([class_idx, x_center, y_center, width, height])

    return yolo_annotations

import cv2
import albumentations as A

# Define the transformation
def crop_n_transform(image, human_bbox, hand_bboxes):
    x1, y1, x2, y2 = human_bbox
    
    bbox_params = A.BboxParams(format='pascal_voc', label_fields=[])
    transform = A.Compose([
        A.Crop(x_min=x1, y_min=y1, x_max=x2, y_max=y2, p=1.0),
    ], bbox_params=bbox_params)
    
    # Convert bounding boxes to the required format
    bboxes = [[hx1, hy1, hx2, hy2] for (hx1, hy1, hx2, hy2) in hand_bboxes]
    
    # Apply the transformation
    transformed = transform(image=image, bboxes=bboxes)
    return transformed['image'], transformed['bboxes']