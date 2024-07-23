import argparse
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import cv2

from modules.target_person_detection import target_person_detection
from modules.crop_target_box import crop_target_box
from modules.gesture_recognition import gesture_recognition
from modules.gesture_synthesis import gesture_synthesis
from prep.select_device import select_device
from prep.load_model import load_targets_model, load_gestures_model
from prep.LoadImages import LoadImages
from prep.inference_prep import img_prep, warmup
from prep.time_synch import time_synchronized
from prep.plots import plot_one_box
from prep.label_mapping import class_to_label
from prep.save_predictions import covert_to_COCO_and_save_json

def main():
    target, img_source, show_plots, save = opt.target, opt.img_source, opt.show_plots, opt.save
    results = {}
    speed = []
    
    # Initialize
    device = select_device('0')
    half = device.type != 'cpu' 
    
    # Load all models and prep everything that I'll need for inference
    targets_model, stride = load_targets_model(target, device, half)
    gestures_model, stride_g = load_gestures_model(device, half)
    
    # Load DataLoader
    dataset = LoadImages(img_source, img_size=1024, stride=stride)
    print(dataset)
    
    # Prep for warmup
    old_img_w = old_img_h = 1024
    old_img_b = 1
    
    # Initialize plot
    if show_plots:
        num_columns = 3
        num_rows = (dataset.nf + num_columns - 1) // num_columns
        fig, axes = plt.subplots(num_rows, num_columns * 2, figsize=(15, 5 * num_rows))
        plt.subplots_adjust(hspace=0.5)
        axes = axes.flatten()
        index = 0
    
    for path, img, im0s in tqdm(dataset):
        img = img_prep(img, device, half)
        warmup(targets_model, device, old_img_b, old_img_h, old_img_w, img)
        
        # Start timer
        t1 = time_synchronized()

        # Detect target person on imgs
        person_bboxs = target_person_detection(targets_model, path, img, im0s, opt.conf_thres)
        
        # Extract
        person_extracted_img, xyxy = crop_target_box(person_bboxs, im0s)
        
        if isinstance(person_extracted_img, int) and person_extracted_img == -1:
            p = -1
            
        else:
            # Detect gestures
            img = img_prep(person_extracted_img, device, half, cropped_img=True, stride=stride_g)
            gestures_preds = gesture_recognition(gestures_model, path, img, person_extracted_img, opt.conf_thres)
            
            # Classify
            p = gesture_synthesis(gestures_preds)
        
        # Stop timer & save results
        t2 = time_synchronized()
        full_time = 1E3 * (t2 - t1)
        speed.append(full_time)
        results[os.path.basename(path)] = int(p)
        
        # Prep subplot
        if show_plots and not isinstance(person_extracted_img, int):
            plot_one_box(xyxy, im0s, color=[0, 255, 0], label='target', line_thickness=8)
            for gesture_pred in gestures_preds:
                label = str(round(gesture_pred[4], 2)) + " " + class_to_label(gesture_pred[5])
                plot_one_box(gesture_pred[0:4], person_extracted_img, color=[0, 0, 255], label=label, line_thickness=4)

            ax1 = axes[index * 2]
            ax1.imshow(cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB))
            ax1.set_title(os.path.basename(path))
            ax1.axis('off')

            ax2 = axes[index * 2 + 1]
            ax2.imshow(cv2.cvtColor(person_extracted_img, cv2.COLOR_BGR2RGB))
            ax2.set_title(f'p={int(p)}')
            ax2.axis('off')

            index += 1

        
    # Print final results
    print(results)
    print(speed)
    average_speed = (sum(speed) / len(speed))
    print(f"average time: {average_speed:.1f}ms")
    
    if show_plots:
        for i in range(index * 2, len(axes)):
            axes[i].axis('off')

        plt.show()
        
    if save:
        covert_to_COCO_and_save_json(target, results)
        with open('./results/speed.txt', 'a') as f:
            f.write(f"Target_{target}: {average_speed}\n") 
        print(f"Average inference time saved to ./results/speed.txt")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=0, help='ID number of the selected participant')
    parser.add_argument('--img-source', type=str, default='', help='path to image')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--show-plots', action='store_true', help='whether the visualizations are printed along with code execution')
    parser.add_argument('--save', action='store_true', help='whether to save the results')
    opt = parser.parse_args()

    main()