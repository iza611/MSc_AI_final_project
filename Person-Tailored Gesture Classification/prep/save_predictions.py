import json

def covert_to_COCO_and_save_json(target, predictions_dict):
    # Load ground truth JSON file
    gt_path = f"./../datasets/SIGGI/full/{target}.json"
    with open(gt_path, 'r') as f:
        coco_data = json.load(f)

    # Extract image IDs and categories
    image_id_mapping = {img['file_name']: img['id'] for img in coco_data['images']}
    category_id_mapping = {cat['name']: cat['id'] for cat in coco_data['categories']}

    # Prepare the COCO format prediction list
    predictions = []

    for img_name, pred_class in predictions_dict.items():
        pred_class = str(pred_class)
        if img_name in image_id_mapping and pred_class in category_id_mapping:
            prediction = {
                "image_id": image_id_mapping[img_name],
                "category_id": category_id_mapping[pred_class]
            }
            predictions.append(prediction)
        else:
            raise Exception(f"Couldnt find image {img_name} or category {pred_class}.")

    # Save predictions in COCO format
    with open(f'./results/predictions_{target}.json', 'w') as f:
        json.dump({"predictions": predictions}, f, indent=4)
    
    print("Saved all predictions in a COCO-compatible JSON file " + f'./results/predictions_{target}.json')
