import os
import csv
import json
import torch
import logging
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser

from tqdm.auto import tqdm

# Grounding DINO
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# GSAM utilities
from grounded_sam_labeler_util import compute_iou, load_gt_mask, to_numpy_image

# Setup logging
FORMAT = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level = logging.INFO, format = FORMAT)
logger = logging.getLogger(__name__)

class GSAMDatasetLabeler:
    
    def __init__(
        self,
        root,
        img_dir,
        gt_dir,
        csv_path,
        out_dir,
        gd_model,
        sam_predictor,
        device,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        iou_threshold: float = 0.75,
        max_images: Optional[int] = None    # for testing purposes
    ) -> None:
        """
        Initializes GSAMDatasetLabeler class. GSAM is used to label custom dataset.

        Args:
            root (Path): Root directory for images and ground truth masks.
            img_dir (Path): Directory containing dataset original images.
            gt_dir (Path): Directory containing dataset ground truth masks.
            csv_path (Path): CSV file path containing dataset metadata.
            out_dir (Path): Output directory for storing results.
            gd_model: Grounding DINO model instance.
            sam_predictor: SAM predictor instance.
            device (torch.device): Device on which to run models.
            box_threshold (float, optional): Confidence threshold for bounding boxes. Default 0.30.
            text_threshold (float, optional): Confidence threshold for text predictions. Default 0.25.
            iou_threshold (float, optional): IoU threshold for keeping masks. Default 0.75.
            max_images (Optional[int], optional): Maximum number of images to process. Default None.
        """
        self.root = root
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.csv_path = csv_path
        self.out_dir = out_dir
        self.gd_model = gd_model
        self.sam_predictor = sam_predictor
        self.device = device
        
        # Thresholds
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold
        self.max_images = max_images
        
        # Directories
        self.kept_dir = out_dir / "kept"
        self.discarded_dir = out_dir / "discarded"
        self.lbl_path = out_dir / "labels.csv"
        
    def create_directories(self):
        """
        Create necessary output directories.
        """
        os.makedirs(self.out_dir, exist_ok = True)
        os.makedirs(self.kept_dir, exist_ok = True)
        os.makedirs(self.discarded_dir, exist_ok = True)
    
    def process_single_image(self, row: pd.Series, lbl_writer: csv.DictWriter) -> bool:
        """
        Processes one single image using GSAM. Grounding DINO is used to predict bounding boxes, 
        SAM to generate masks. Generated masks are then compared against the ground truth. If a 
        GSAM mask with IoU greater than iou value specified as parameter against the Ground 
        Truth exists, the current image is kept. Otherwise it is temporarily discarded.

        Args:
            row (pd.Series): Row from CSV file containing image information.
            lbl_writer (csv.DictWriter): CSV writer for saving label information.

        Returns:
            bool: True if the image was successfully processed. Otherwise False.
        """
        image_name = row['image_name']
        class_name = row['target_type']
        
        logger.info(f"\nProcessing '{image_name}' (class '{class_name}')")
        
        try:
            # 1. Load image to process
            image_path = self.img_dir / image_name
            if not image_path.exists():
                logger.warning(f"Image not found: '{image_path}'")
                return False
            
            image, image_transformed = load_image(str(image_path))  # np.array, torch.Tensor
            height, width, _ = image.shape
            
            # 2. Run Grounding DINO
            boxes, logits, phrases = predict(
                model = self.gd_model,
                image = image_transformed,
                caption = class_name,
                box_threshold = self.box_threshold,
                text_threshold = self.text_threshold
            )
            
            # 3. Load ground truth mask and convert it in binary
            gt_path = load_gt_mask(self.gt_dir, image_name)
            if gt_path is None:
                logger.warning(f"Ground truth mask not found for '{image_name}'")
                return False
            
            gt_mask = Image.open(gt_path).convert(mode = "L")
            gt_mask_np = np.array(gt_mask)
            # print(np.unique(gt_mask_np))
            gt_mask_bin = (gt_mask_np > 127).astype(np.uint8)
            
            # 4. Check detections
            if boxes is None or len(boxes) == 0:
                logger.warning(f"No Grounding DINO detections for '{image_name}'")
                return False
            
            # 5. Run segmentation model
            self.sam_predictor.set_image(image)
            
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([width, height, width, height])   # from cxcywh format to xyxy
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image.shape[:2]).to(self.device)
            
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
            
            # 6. Evaluate masks and find the best
            masks_info = []
            
            for i, mask_tensor in enumerate(masks):
                mask_np = mask_tensor[0].detach().cpu().numpy()
                iou = compute_iou(mask_np, gt_mask_bin)
                
                masks_info.append({
                    "index": i,
                    "box": boxes_xyxy[i].cpu().numpy(),
                    "mask": mask_np,
                    "iou": iou,
                    "phrase": phrases[i] if i < len(phrases) else class_name,
                    "logit": logits[i].item() if i < len(logits) else 0.0
                })
            
            best = max(masks_info, key = lambda x: x["iou"])
            base_name = Path(image_name).stem   # without extension
            
            # 7. Save results
            is_kept = best["iou"] >= self.iou_threshold
            target_dir = self.kept_dir if is_kept else self.discarded_dir
            out_image_dir = target_dir / base_name
            os.makedirs(out_image_dir, exist_ok = True)
            
            # 7a. save original image and corresponding ground truth for reference
            Image.fromarray(to_numpy_image(image)).save(out_image_dir / f"{base_name}__img.png")
            Image.fromarray((gt_mask_bin * 255).astype(np.uint8)).save(out_image_dir / f"{base_name}__gt.png")
            
            for info in masks_info:
                is_odd = (info["index"] == best["index"]) and is_kept
                
                # 7b. save masks odd and normal
                mask_suffix = "__ODD" if is_odd else ""
                mask_filename = f"{base_name}__mask_box{info['index']}{mask_suffix}.png"
                mask_path = out_image_dir / mask_filename
                Image.fromarray((info["mask"] * 255).astype(np.uint8)).save(mask_path)
                
                # 7c. write CSV file
                lbl_writer.writerow({
                    "image_name": image_name,
                    "mask_filename": str(mask_path.relative_to(self.out_dir)),
                    "is_odd": int(is_odd),
                    "is_kept": int(is_kept),
                    "iou": f"{info['iou']:.3f}",
                    "confidence": f"{info['logit']:.3f}",
                    "category": info["phrase"]
                })
            
            # 8. Log result
            if is_kept:
                logger.info(f" KEPT - best IoU = {best['iou']:.3f} (mask {best['index']} marked as ODD)")
            else:
                logger.info(f" DISCARDED - best IoU = {best['iou']:.3f} < {self.iou_threshold}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing '{image_name}': {str(e)}")
            return False
    
    def run(self):
        """
        Run the GSAM dataset labeling.
        """
        logger.info('\n========== GSAM LABELING ==========')

        # Setup output directories
        self.create_directories()
        
        # Load CSV
        img_props = pd.read_csv(self.csv_path, sep = ",")
        total_images = len(img_props) if self.max_images is None else min(len(img_props), self.max_images)
        logger.info(f"Processing {total_images} images from {len(img_props)} total.")
        
        # Process images
        with open(self.lbl_path, mode = 'w', newline = '') as lbl_file:
            lbl_writer = csv.DictWriter(lbl_file, fieldnames=[
                "image_name", "mask_filename", "is_odd", "is_kept", 
                "iou", "confidence", "category"
            ])
            lbl_writer.writeheader()
        
            for index, row in tqdm(img_props.iloc[:total_images].iterrows(), 
                total=total_images, desc="GSAM Labeling Progress"):
                self.process_single_image(row, lbl_writer) 

        logger.info('\n========== GSAM LABELING FINISHED ==========')
        logger.info(f"Results of labeling saved in {self.out_dir}")   

def main(args):

    logger.info('\n========== GSAM LABELING ==========')
    
    # Initialize and run labeler
    labeler = GSAMDatasetLabeler(
        root=args.root_dir,
        img_dir=args.img_dir,
        gt_dir=args.gt_dir,
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        gd_model=args.gd_model,  
        sam_predictor=args.sam_predictor, 
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        iou_threshold=args.iou_threshold,
        max_images=args.max_images
    )
    
    labeler.run()
    
    logger.info('========== GSAM LABELING FINISHED ==========')
    logger.info(f"Results of labeling saved in {args.out_dir}")


# For future uses with CLI
# TODO: gd_model and sam_predictor arguments must be pass as path or str.
# The script must implement a function (e.g. load_model) to load actual
# model instances of GroundingDINO and SAM, using the provided paths.
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--root-dir', type=Path, default='/content/O3_data')
    parser.add_argument('--out-dir', type=Path, default='/content/O3_output')
    parser.add_argument('--img-dir', type=Path, default=None)
    parser.add_argument('--gt-dir', type=Path, default=None)
    parser.add_argument('--csv-path', type=Path, default=None)
    parser.add_argument('--gd-model', required=True)    # pass as path or str
    parser.add_argument('--sam-predictor', required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--box-threshold', type=float, default=0.30)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument('--iou-threshold', type=float, default=0.75)
    parser.add_argument('--max-images', type=int, default=None, 
                        help='Maximum number of images to process (for testing purposes)')
    
    args = parser.parse_args()

    args.device = torch.device(args.device)

    if args.img_dir is None:
        args.img_dir = args.root_dir / 'images'

    if args.gt_dir is None:
        args.gt_dir = args.root_dir / 'targ_labels'

    if args.csv_path is None:
        args.csv_path = args.root_dir / 'image_properties.csv'

    main(args)