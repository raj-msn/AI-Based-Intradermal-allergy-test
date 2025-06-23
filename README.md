# Intradermal Allergy Test Analysis

> Designed a DL pipeline for analyzing intradermal allergy test images using object detection and instance segmentation models (DETR, YOLO, Detectron2). Employed StyleGAN2 ADA and augmentation to overcome limited data. Achieved 93% mAP and deployed via web app. Currently optimizing for edge deployment and working for academic publication.

This repository contains the complete pipeline for the project described above. It addresses the challenge of limited medical imaging data by first using a generative model (StyleGAN2-ADA) for data augmentation, and then training and evaluating multiple state-of-the-art object detection models on the enriched dataset.

---

## The Story of This Project

1.  **The Challenge**: Intradermal allergy tests produce images that can be difficult to analyze consistently. A key bottleneck in applying deep learning is often the scarcity of high-quality, annotated medical images.

2.  **Stage 1: Data Augmentation with StyleGAN**: To overcome this, the first part of our pipeline involves training a generative model on the limited dataset. This StyleGAN learns the underlying distribution of the images and can generate an unlimited number of new, synthetic training examples.

3.  **Stage 2: Training Detection Models**: With a robust dataset of original and synthetic images, we then train several powerful object detection and instance segmentation models (YOLOv8, DETR, Faster R-CNN, Detectron2) to accurately identify the "flares" and "tags" in the images.

4.  **Stage 3: Inference and Evaluation**: Finally, we use the best-performing trained model to run inference on new, unseen images, and provide helper scripts to evaluate their performance.

---

## Project Structure

The repository is organized into a modular and reusable structure:

```
├── configs/                # Configuration files (e.g., data.yaml for YOLO)
├── data/                   # Raw and augmented image data (ignored by Git)
├── notebooks/              # Original Jupyter notebooks for exploration and analysis
├── src/                    # All source code
│   ├── data_utils.py       # PyTorch Dataset classes
│   ├── detr_helpers.py     # Helper functions for DETR model
│   ├── detectron_helpers.py# Helper functions for Detectron2
│   ├── faster_rcnn_helpers.py # Helper functions for Faster R-CNN
│   ├── stylegan_helpers.py # Helper functions for the StyleGAN pipeline
│   ├── yolo_helpers.py     # Helper functions for YOLOv8
│   ├── train.py            # Main script to train detection models
│   ├── inference.py        # Main script to run inference
│   ├── run_stylegan.py     # Dedicated script for the StyleGAN workflow
│   └── visualize.py        # Scripts for creating result plots
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    # Detectron2 often requires a separate installation step
    # pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
    ```

---

## How to Use This Repository

The project is designed to be run from the command line using the scripts in the `src/` directory.

### Step 1: Generate Augmented Data with StyleGAN (Optional)

If you need to augment your dataset, use the `src/run_stylegan.py` script. It has several actions.

1.  **Prepare your raw images** (resize and format them):
    ```bash
    python src/run_stylegan.py prepare --input-dir /path/to/raw/images --output-dir /path/to/prepared/images
    ```
2.  **Create the zipped dataset** required by StyleGAN:
    ```bash
    python src/run_stylegan.py create_dataset --input-dir /path/to/prepared/images --zip-path /path/to/dataset.zip
    ```
3.  **Train the StyleGAN model**:
    ```bash
    python src/run_stylegan.py train --zip-path /path/to/dataset.zip --output-dir /path/to/stylegan/results
    ```
4.  **Generate new images**:
    ```bash
    python src/run_stylegan.py generate --weights /path/to/stylegan/results/network-snapshot.pkl --output-dir /path/to/generated/images
    ```

### Step 2: Train a Detection Model

Use the `src/train.py` script to train one of the detection models on your (now augmented) dataset.

*   **To train YOLOv8:**
    ```bash
    python src/train.py --model yolo --epochs 100 --yolo-data /path/to/your/data.yaml
    ```
*   **To train DETR or Faster R-CNN** (using a COCO-formatted dataset):
    ```bash
    python src/train.py --model detr --epochs 50 --coco-data /path/to/coco/dataset
    ```
*   **To train Detectron2:**
    ```bash
    python src/train.py --model detectron2 --epochs 20 --detectron-data /path/to/detectron/dataset
    ```

### Step 3: Run Inference

Use the `src/inference.py` script to make predictions on new images using your trained model weights.

*   **Example with a trained YOLO model:**
    ```bash
    python src/inference.py --model yolo --weights /path/to/yolo/weights.pt --input /path/to/new/images --output /path/to/save/results
    ```
*   **Example with a trained DETR model:**
    ```bash
    python src/inference.py --model detr --weights /path/to/detr.ckpt --num-classes 3 --input /path/to/new/images --output /path/to/save/results
    ```

---

### Exploring the Original Notebooks

The `notebooks/` directory contains the original, raw experiments and analyses. These are great for understanding the thought process and seeing the step-by-step development that led to the final, polished scripts (coded on Google Collab).  
