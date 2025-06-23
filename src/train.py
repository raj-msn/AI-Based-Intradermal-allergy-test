import argparse
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrImageProcessor

# Import helper functions and classes from our new modules
from data_utils import CocoDetection, CocoDetectionFasterRCNN
from detr_helpers import Detr, collate_fn as detr_collate_fn
from faster_rcnn_helpers import FasterRCNNLightning, get_faster_rcnn_model
from yolo_helpers import train_yolo
from detectron_helpers import setup_detectron_cfg, train_detectron_model, register_coco_dataset

def main(args):
    """Main training function based on the selected model."""

    if args.model == 'yolo':
        print("Training YOLOv8...")
        if not args.yolo_data:
            print("Error: --yolo-data (path to data.yaml) is required for YOLOv8 training.")
            return
        train_yolo(args.yolo_data, epochs=args.epochs, project_name="YOLOv8_Training")
        print("YOLOv8 training complete.")

    elif args.model == 'detr':
        print("Training DETR...")
        if not args.coco_data:
            print("Error: --coco-data (path to COCO dataset root) is required for DETR training.")
            return
        image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
        train_dir = os.path.join(args.coco_data, "train")
        val_dir = os.path.join(args.coco_data, "valid")
        
        train_dataset = CocoDetection(image_directory_path=train_dir, image_processor=image_processor)
        val_dataset = CocoDetection(image_directory_path=val_dir, image_processor=image_processor, train=False)
        
        num_labels = len(train_dataset.coco.cats)

        train_loader = DataLoader(dataset=train_dataset, collate_fn=detr_collate_fn, batch_size=4, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, collate_fn=detr_collate_fn, batch_size=4)
        
        model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, num_labels=num_labels)
        trainer = pl.Trainer(max_epochs=args.epochs, gradient_clip_val=0.1, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
        trainer.fit(model, train_loader, val_loader)
        print("DETR training complete.")

    elif args.model == 'faster-rcnn':
        print("Training Faster R-CNN...")
        if not args.coco_data:
            print("Error: --coco-data (path to COCO dataset root) is required for Faster R-CNN training.")
            return
        train_dir = os.path.join(args.coco_data, "train")
        
        train_dataset = CocoDetectionFasterRCNN(image_directory_path=train_dir, train=True)
        num_classes = len(train_dataset.coco.cats) + 1
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

        frcnn_model = get_faster_rcnn_model(num_classes)
        model = FasterRCNNLightning(frcnn_model, lr=1e-4)
        
        trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
        trainer.fit(model, train_loader)
        print("Faster R-CNN training complete.")

    elif args.model == 'detectron2':
        print("Training Detectron2...")
        if not args.detectron_data:
            print("Error: Please provide the path to the Detectron2 dataset using --detectron-data")
            return
            
        train_set_name = "d2_train"
        test_set_name = "d2_test"
        
        register_coco_dataset(train_set_name, os.path.join(args.detectron_data, "train", "_annotations.coco.json"), os.path.join(args.detectron_data, "train"))
        register_coco_dataset(test_set_name, os.path.join(args.detectron_data, "test", "_annotations.coco.json"), os.path.join(args.detectron_data, "test"))

        cfg = setup_detectron_cfg()
        output_dir = "detectron_training_output" # Made relative
        train_detectron_model(cfg, train_set_name, test_set_name, output_dir, max_iter=args.epochs * 100) # Convert epochs to iterations
        print("Detectron2 training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main training script for Intradermal Test project.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['yolo', 'detr', 'faster-rcnn', 'detectron2'],
                        help="The model to train.")
    parser.add_argument('--epochs', type=int, default=100, 
                        help="Number of training epochs (approximated for Detectron2).")
    # --- Dataset paths ---
    parser.add_argument('--yolo-data', type=str, 
                        help="[YOLO] Path to the data.yaml file.")
    parser.add_argument('--coco-data', type=str, 
                        help="[DETR, FasterRCNN] Path to the root of the COCO-formatted dataset.")
    parser.add_argument('--detectron-data', type=str, 
                        help="[Detectron2] Path to the root of the Detectron2 dataset (e.g., from Roboflow).")

    args = parser.parse_args()
    main(args)