
results = [] # todo: change for results from the model

for result in results:
    boxes = result.boxes.xywh  # Get bbox [x_center, y_center, width, height]
    confidences = result.boxes.conf  # Get confidence scores
    classes = result.boxes.cls  # Get class IDs

    # Save to a TXT file (same name as image)
    label_file = image_path.replace(".jpg", ".txt")  # Adjust for your dataset
    with open(label_file, "w") as f:
        for box, conf, cls in zip(boxes, confidences, classes):
            x, y, w, h = box.tolist()
            class_id = int(cls.item())
            f.write(f"{class_id} {x} {y} {w} {h} {conf:.4f}\n")