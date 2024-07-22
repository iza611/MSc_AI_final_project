import argparse
from modules import target_person_detection, crop_target_box, gesture_recognition, gesture_synthesis
from select_device import select_device
from load_model import load_targets_model, load_gestures_model
from LoadImages import LoadImages
from inference_prep import img_prep, warmup

def main():
    target, img_source, show_plots, save = opt.target, opt.img_source, opt.show_plots, opt.save
    results = {}
    speed = []
    
    # Initialize
    device = select_device('0')
    half = device.type != 'cpu' 
    
    # Load all models and prep everything that I'll need for inference
    targets_model, stride = load_targets_model(target, device, half)
    gestures_model = load_gestures_model(device, half)
    
    # Load DataLoader
    dataset = LoadImages(source, img_size=1024, stride=stride)
    
    old_img_w = old_img_h = 1024
    old_img_b = 1
    
    for path, img, im0s in tqdm(dataset):
        img = img_prep(img, device, half)
        warmup(model, device, old_img_b, old_img_h, old_img_w, img)
        
        # Start timer
        t1 = time_synchronized()

        # Detect target person on imgs
        person_bboxs = target_person_detection(gestures_model, path, img, im0s, opt.conf_thres)
        t2 = time_synchronized()
        
        # Extract
        person_extracted_img = crop_target_box(person_bboxs, img, im0s)
        t3 = time_synchronized()
        
        if (person_extracted_img != -1):
            # Detect gestures
            img = img_prep(person_extracted_img, device, half)
            gestures_preds = gesture_recognition(gestures_model, path, img, person_extracted_img, opt.conf_thres)
            t4 = time_synchronized()
            
            # Classify
            p = gesture_synthesis(gestures_preds)
            t5 = time_synchronized()
            
        else:
            p = -1
            t6 = time_synchronized()
        
        # Calculate speed
        if t5:
            full_time = 1E3 * (t5 - t1)
            p_det_time = 1E3 * (t2 - t1)
            p_extr_time = 1E3 * (t3 - t2)
            g_det_time = 1E3 * (t4 - t3)
            g_cls_time = 1E3 * (t5 - t4)
        else:
            full_time = 1E3 * (t6 - t1)
            p_det_time = 1E3 * (t2 - t1)
            p_extr_time = 1E3 * (t6 - t2)
        
        speed.append(full_time)
        results[path] = p
        
        
    # Print final results
    print(p)
    
#         # Display plots if enabled
#         if show_plots:
#             # TODO: add plotting where each row: |original | target detected | cut | gesture detected | prediction p |
#         # Save results if enabled
        
#     if(save):
#         # TODO: add saving
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=0, help='ID number of the selected participant')
    parser.add_argument('--img-source', type=str, default='', help='path to image')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--show-plots', action='store_true', help='whether the visualizations are printed along with code execution')
    parser.add_argument('--save', action='store_true', help='whether to save the results')
    opt = parser.parse_args()

    main()