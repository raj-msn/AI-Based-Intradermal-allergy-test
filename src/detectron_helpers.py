import os
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
import cv2
import glob
from google.colab.patches import cv2_imshow # Specific to Colab

def register_coco_dataset(dataset_name, json_file, image_root):
    """
    Register a COCO dataset in Detectron2.
    """
    register_coco_instances(dataset_name, {}, json_file, image_root)

def setup_detectron_cfg(weights_path=None, num_classes=3, score_thresh=0.7, architecture="mask_rcnn_R_101_FPN_3x"):
    """
    Create and configure a Detectron2 config object.
    """
    cfg = get_cfg()
    config_file = f"COCO-InstanceSegmentation/{architecture}.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    
    if weights_path:
        cfg.MODEL.WEIGHTS = weights_path
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    return cfg

def train_detectron_model(cfg, train_dataset_name, test_dataset_name, output_dir, max_iter=2000, base_lr=0.001):
    """
    Train a Detectron2 model.
    """
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    return trainer

def run_inference_on_image(predictor, image_path, metadata):
    """
    Run inference on a single image and visualize the output.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1],
                   metadata=metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = out.get_image()[:, :, ::-1]
    
    # Display the image (Colab-specific)
    cv2_imshow(result_image)

    return outputs, result_image

def process_image_folder(predictor, image_folder, output_folder, metadata):
    """
    Run inference on a folder of images and save the results.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder, "*.jpeg")) + \
                  glob.glob(os.path.join(image_folder, "*.png"))

    for image_path in image_paths:
        try:
            _, result_image = run_inference_on_image(predictor, image_path, metadata)
            output_path = os.path.join(output_folder, f"result_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, result_image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}") 