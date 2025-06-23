import torch
import torchvision
import os
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torchvision.transforms import functional as F


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        # Load image
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Image loading
        img_path = os.path.join(self.root, self.coco.imgs[img_id]['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Convert the image to a tensor
        img = F.to_tensor(img)

        # Extracting bounding boxes and masks
        boxes = []
        masks = []
        labels = []
        for ann in annotations:
            # Extract bounding box
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])

            # Extract mask
            mask = None
            if 'segmentation' in ann and ann['segmentation']:
                try:
                    # Convert polygons to binary masks
                    if isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                        # Polygon -- a single object might consist of multiple parts
                        rles = coco_mask.frPyObjects(ann['segmentation'], img.shape[1], img.shape[2])
                        mask = coco_mask.decode(rles)
                        mask = mask[..., 0] if len(mask.shape) > 2 else mask
                    elif isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']:
                        # Handle RLE and uncompressed RLE
                        mask = coco_mask.decode(ann['segmentation'])
                except Exception as e:
                    print(f"Error decoding mask for annotation {ann}: {e}")

                if mask is not None:
                    masks.append(mask)

            # Skip annotations with invalid or missing masks
            if mask is not None:
                labels.append(ann['category_id'])

        # Convert everything into torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks

        return img, target

    def __len__(self):
        return len(self.ids)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        image_processor,
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, "_annotations.coco.json")
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


class CocoDetectionFasterRCNN(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        transforms=None,
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, "_annotations.coco.json")
        super(CocoDetectionFasterRCNN, self).__init__(image_directory_path, annotation_file_path)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetectionFasterRCNN, self).__getitem__(idx)
        image_id = self.ids[idx]

        boxes = []
        labels = []
        area = []
        iscrowd = []

        for obj in target:
            bbox = obj['bbox']
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(obj['category_id'])
            area.append(obj['area'])
            iscrowd.append(obj['iscrowd'])

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([image_id]),
            'area': torch.as_tensor(area, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64)
        }


        if self.transforms is not None:
            img, target = self.transforms(img, target)


        return img, target 