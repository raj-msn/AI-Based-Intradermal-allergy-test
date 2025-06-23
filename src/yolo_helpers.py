import os
import yaml
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow


def train_yolo(data_yaml_path, epochs, project_name, model_name="yolov8n.yaml"):
    """
    Train a YOLOv8 model.
    """
    model = YOLO(model_name)
    results = model.train(data=data_yaml_path, epochs=epochs, project=project_name)
    return results, model.trainer.best


def visualize_yolo_predictions(model_weights, image_dir):
    """
    Visualize predictions from a trained YOLO model on a directory of images.
    """
    model = YOLO(model_weights)
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    for image_path in image_files:
        results = model(image_path)
        res_plotted = results[0].plot()
        cv2_imshow(res_plotted) # This is specific to Google Colab


def get_bounding_box_from_polygon(polygon_points, image_width, image_height):
    """Convert polygon points to bounding box with better normalization"""
    x_coords = []
    y_coords = []

    for i in range(0, len(polygon_points), 2):
        x = polygon_points[i] * image_width
        y = polygon_points[i + 1] * image_height
        x_coords.append(x)
        y_coords.append(y)

    x1 = min(x_coords)
    y1 = min(y_coords)
    x2 = max(x_coords)
    y2 = max(y_coords)

    return [x1, y1, x2, y2]

def parse_yolo_annotations(line, width, height):
    """Parse YOLO format annotations directly from the line"""
    values = [float(x) for x in line.strip().split()]
    class_id = int(values[0])
    coordinates = values[1:]
    bbox = get_bounding_box_from_polygon(coordinates, width, height)
    return class_id, bbox

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box1[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def evaluate_yolo_model(model_weights, data_yaml_path, val_images_dir, val_labels_dir, iou_threshold=0.25):
    """
    Custom evaluation for a YOLO model.
    """
    model = YOLO(model_weights)
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    class_names = data_config['names']
    num_classes = len(class_names)

    all_predictions = []
    all_ground_truths = []

    for image_file in os.listdir(val_images_dir):
        if not image_file.endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(val_images_dir, image_file)
        img = Image.open(image_path)
        width, height = img.size

        results = model(image_path)
        image_predictions = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                image_predictions.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id
                })

        label_file = os.path.join(val_labels_dir, image_file.rsplit('.', 1)[0] + '.txt')
        image_ground_truths = []

        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        class_id, bbox = parse_yolo_annotations(line, width, height)
                        image_ground_truths.append({
                            'bbox': bbox,
                            'class_id': class_id
                        })
                    except Exception as e:
                        print(f"Error processing line: {line}. Error: {e}")
                        continue

        all_predictions.append(image_predictions)
        all_ground_truths.append(image_ground_truths)

    class_metrics = {}
    for class_id in range(num_classes):
        true_positives = []
        false_positives = []
        scores = []
        num_ground_truths = sum(sum(1 for gt in img_gts if gt['class_id'] == class_id) for img_gts in all_ground_truths)

        for img_preds, img_gts in zip(all_predictions, all_ground_truths):
            class_preds = [pred for pred in img_preds if pred['class_id'] == class_id]
            class_preds.sort(key=lambda x: x['confidence'], reverse=True)
            matched_gts = set()

            for pred in class_preds:
                scores.append(pred['confidence'])
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(img_gts):
                    if gt['class_id'] != class_id or gt_idx in matched_gts:
                        continue
                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx not in matched_gts:
                    true_positives.append(1)
                    false_positives.append(0)
                    matched_gts.add(best_gt_idx)
                else:
                    true_positives.append(0)
                    false_positives.append(1)

        if num_ground_truths > 0 and len(true_positives) > 0:
            true_positives = np.array(true_positives)
            false_positives = np.array(false_positives)

            cum_tp = np.cumsum(true_positives)
            cum_fp = np.cumsum(false_positives)

            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / num_ground_truths

            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap += p / 11
            
            class_metrics[class_names[class_id]] = {
                'AP': ap,
                'Precision': precision[-1] if len(precision) > 0 else 0,
                'Recall': recall[-1] if len(recall) > 0 else 0
            }
        else:
            class_metrics[class_names[class_id]] = {'AP': 0, 'Precision': 0, 'Recall': 0}

    mean_ap = np.mean([metrics['AP'] for metrics in class_metrics.values()])
    mean_precision = np.mean([metrics['Precision'] for metrics in class_metrics.values()])
    mean_recall = np.mean([metrics['Recall'] for metrics in class_metrics.values()])

    return {
        'class_metrics': class_metrics,
        'mAP50': mean_ap,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall
    } 