from pathlib import Path

def crop_target_box(person_bboxs, dataset):
    
    person_extracted_imgs ={}
    
    for path, img, im0s, vid_cap in tqdm(dataset):
        path = Path(path)
        x1, y1, x2, y2 = person_bboxs[path.name]  # single bbx 'xyxy'
        cropped_image = im0s[y1:y2, x1:x2]
        person_extracted_imgs[path.name] = cropped_image
        
    return person_extracted_imgs
        

        