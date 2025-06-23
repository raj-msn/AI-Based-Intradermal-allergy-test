import argparse
import os
import torch
import cv2
import glob
from PIL import Image
import supervision as sv

# Import helpers
from ultralytics import YOLO
from transformers import DetrImageProcessor
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

from detr_helpers import Detr
from detectron_helpers import setup_detectron_cfg

def run_yolo_inference(weights_path, input_dir, output_dir):
    """Run inference with a YOLOv8 model."""
    print(f"Running YOLOv8 inference with weights {weights_path}...")
    model = YOLO(weights_path)
    image_paths = glob.glob(os.path.join(input_dir, '*.[jp][pn][g]'))
    
    for img_path in image_paths:
        results = model(img_path)
        annotated_image = results[0].plot()
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, annotated_image)
    print(f"YOLOv8 inference complete. Results saved to {output_dir}")

def run_detectron_inference(weights_path, input_dir, output_dir):
    """Run inference with a Detectron2 model."""
    print(f"Running Detectron2 inference with weights {weights_path}...")
    # This requires dataset registration to get metadata, which is a bit of a workaround for inference.
    # A better approach would be to save/load metadata separately.

    try:
        metadata = MetadataCatalog.get("special_project_is-1-train")
    except KeyError:
        print("Warning: Metadata not found. Using default COCO metadata. Class names might be incorrect.")
        print("For accurate labels, ensure you register your dataset first or modify this script to load metadata.")
        from detectron2.data.datasets import register_coco_instances
        # Dummy registration to get default metadata
        register_coco_instances("dummy_coco", {}, "dummy.json", "dummy_dir")
        metadata = MetadataCatalog.get("dummy_coco")


    cfg = setup_detectron_cfg(weights_path=weights_path)
    predictor = DefaultPredictor(cfg)
    image_paths = glob.glob(os.path.join(input_dir, '*.[jp][pn][g]'))
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        outputs = predictor(img)
        v = sv.Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
        annotated_image = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, annotated_image)
    print(f"Detectron2 inference complete. Results saved to {output_dir}")


def run_detr_inference(weights_path, input_dir, output_dir, num_labels):
    """Run inference with a DETR model."""
    print(f"Running DETR inference with weights {weights_path}...")
    model = Detr.load_from_checkpoint(weights_path, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, num_labels=num_labels)
    model.eval()
    image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    box_annotator = sv.BoxAnnotator()
    image_paths = glob.glob(os.path.join(input_dir, '*.[jp][pn][g]'))

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        encoding = image_processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"]
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=None)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        detections = sv.Detections.from_transformers(transformers_results=results)
        
        annotated_image = box_annotator.annotate(
            scene=cv2.imread(img_path), 
            detections=detections,
            labels=[model.config.id2label[label.item()] for label in detections.class_id]
        )
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, annotated_image)
    print(f"DETR inference complete. Results saved to {output_dir}")


def main(args):
    os.makedirs(args.output, exist_ok=True)
    
    if args.model == 'yolo':
        run_yolo_inference(args.weights, args.input, args.output)
    elif args.model == 'detectron2':
        run_detectron_inference(args.weights, args.input, args.output)
    elif args.model == 'detr':
        if not args.num_classes:
            print("Error: --num-classes is required for DETR model.")
            return
        run_detr_inference(args.weights, args.input, args.output, args.num_classes)
    else:
        print(f"Model {args.model} not implemented for inference yet.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main inference script for Intradermal Test project.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['yolo', 'detectron2', 'detr'],
                        help="The model to use for inference.")
    parser.add_argument('--weights', type=str, required=True,
                        help="Path to the trained model weights (.pt, .pth, .ckpt, or .pkl).")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the input image or directory of images.")
    parser.add_argument('--output', type=str, default="inference_results",
                        help="Path to the directory to save output images.")
    # Model-specific arguments
    parser.add_argument('--num-classes', type=int,
                        help="[DETR only] The number of classes the model was trained on.")
                        
    args = parser.parse_args()
    main(args) 