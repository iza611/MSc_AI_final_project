import argparse
from modules import target_person_detection, crop_target_box, gesture_recognition, gesture_synthesis

LoadImages

def main():
    target, img_source, show_plots, save = opt.target, opt.img_source, opt.show_plots, opt.save
    
    # Set DataLoader
    dataset = LoadImages(img_source, img_size=imgsz, stride=stride)
    
    # Initialise results dictionary {img_name : gesture_prediction}
    p = {}
    
    # Detect target person on all provided imgs and save to dict {img_name : person_bbox}
    person_bboxs = target_person_detection(target, dataset)
    
    # If no target present set gesture_prediction to -1 & remove element from person_bboxs dict
    no_target_imgs = [img_name for img_name, bbxs in person_bboxs.items() if len(bbxs) == 0]
    for img_name in no_target_imgs:
        p[img_name] = -1
        del person_bboxs[img_name]
    
    # Crop, recognise gestures, synthesise 
    person_extracted_imgs = crop_target_box(person_bboxs, dataset)
    gestures_bboxs, gestures_classes = gesture_recognition(person_extracted_imgs)
    p = gesture_synthesis(gestures_classes)
    
    # Print final results
    for img_name, gesture_prediction in p.items():
        print(f"{img_name}: {gesture_prediction}")
    
    # Display plots if enabled
    if show_plots:
        # TODO: add plotting where each row: |original | target detected | cut | gesture detected | prediction p |
        
    # Save results if enabled
    if(save):
        # TODO: add saving
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=0, help='ID number of the selected participant')
    parser.add_argument('--img-source', type=str, default='', help='path to image')
    parser.add_argument('--show-plots', action='store_true', help='whether the visualizations are printed along with code execution')
    parser.add_argument('--save', action='store_true', help='whether to save the results')
    opt = parser.parse_args()

    main()