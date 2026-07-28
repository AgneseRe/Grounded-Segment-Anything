"""Pipeline for automatic labeling using Grounding DINO + SAM (GSAM)."""

import os
import csv
import torch
import logging
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from torchvision import models
from typing import Optional, Tuple
from argparse import ArgumentParser

from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.inference import apply_nms, load_image, predict
from grounded_sam_labeler_util import compute_iou, keep_valid_boxes, load_gt_mask, to_numpy_image

# Setup logging
FORMAT = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level = logging.INFO, format = FORMAT)
logger = logging.getLogger(__name__)

def load_resnet_encoder(device: torch.device) -> torch.nn.Module:
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) 
    encoder = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # remove last FC layer
    encoder.to(device).eval()
    return encoder

def apply_nms_on_masks(masks_info: list, iou_threshold: float = 0.2) -> list:

    if len(masks_info) <= 1:
        return masks_info
    
    masks_info = sorted(masks_info, key=lambda x: x['logit'], reverse=True)

    keep = []
    suppressed = set()

    for i, info in enumerate(masks_info):
        if i in suppressed:  # already suppressed
            continue
        keep.append(info)
        mask_i = info['mask']
        # discard all following masks with overlap greater than threshold
        for j in range(i+1, len(masks_info)):
            if j in suppressed:
                continue
            mask_j = masks_info[j]['mask']
            # compute iou between masks
            iou = compute_iou(mask_i, mask_j)
            if iou > iou_threshold:
                suppressed.add(j)
                logger.info(f'Mask {j} suppressed by mask {i} (IoU = {iou:.4f})')
    
    return keep

def compute_coverage_pollution_distractors(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """
    Computes coverage and pollution metrics for distractors.

    Args:
        pred_mask (np.ndarray): Binary union of distractors masks generated with GSAM.
        gt_mask (np.ndarray): Ground truth mask of distractors.

    Returns:
        Tuple[float, float]: Coverage and pollution of distractors masks.
    """
    pred_mask_bool = pred_mask.astype(bool)
    gt_mask_bool = gt_mask.astype(bool)
    # Compute masks area and intersection
    pred_mask_area = np.sum(pred_mask_bool)
    gt_mask_area = np.sum(gt_mask_bool)
    intersection_area = np.sum(np.logical_and(pred_mask_bool, gt_mask_bool))
    # Coverage: how much of ground truth is covered by GSAM predictions
    coverage = intersection_area / gt_mask_area if gt_mask_area > 0 else 0.0 
    # Pollution: how much of GSAM predictions is outside the ground truth
    pollution = (pred_mask_area - intersection_area) / pred_mask_area if pred_mask_area > 0 else 0.0

    return coverage, pollution


class GSAMDatasetLabeler:

    def __init__(
        self,
        root,
        img_dir,
        gt_odd_dir,
        gt_dist_dir,
        csv_path,
        out_dir,
        gd_model,
        sam_predictor,
        device,
        dataset,   # O3 or FLUX scenes
        csv_obj_desc_path: str = None,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        iou_threshold: float = 0.75,
        nms_threshold: Optional[float] = None,
        ada_box_threshold: bool = False,  
        ada_nms_threshold: bool = False, 
        nms_strategy: str = "num_distractors",
        penalty_score: float = 1e-4,
        max_images: Optional[int] = None,   # for testing purposes
    ) -> None:
        """
        Initializes GSAMDatasetLabeler class. GSAM is used to label custom dataset.

        Args:
            root (Path): Root directory for images and ground truth masks.
            img_dir (Path): Directory containing dataset original images.
            gt_odd_dir (Path): Directory containing dataset ground truth masks for odd object.
            gt_dist_dir (Path): Directory containing dataset ground truth masks for distractors.
            csv_path (Path): CSV file containing dataset metadata.
            csv_obj_desc_path (str): CSV file containing object description for FLUX scenes. 
            out_dir (Path): Output directory for storing results.
            gd_model: Grounding DINO model instance.
            sam_predictor: SAM predictor instance.
            device (torch.device): Device on which to run models.
            box_threshold (float, optional): Confidence threshold for bounding boxes. Default 0.30.
            text_threshold (float, optional): Confidence threshold for text predictions. Default 0.25.
            iou_threshold (float, optional): IoU threshold for keeping masks. Default 0.75.
            nms_threshold (float, optional): IoU threshold for NMS. Default None.
            ada_box_threshold (bool): If True, box_threshold depends on number of distractors in image. 
                If False, same box_threshold for all processed images. Default False.
            ada_nms_threshold (bool): If True, nms_threshold depends on nms_strategy. Default True.
            nms_strategy (str): It can be num_distractors, logits_variance or boxes_overlap. Default num_distractors.
            max_images (int, optional): Maximum number of images to process. Default None.

            NMS, adaptive threshold, non-overlapping masks for better results.
        """
        self.root = root
        self.img_dir = img_dir
        self.gt_odd_dir = gt_odd_dir
        self.gt_dist_dir = gt_dist_dir
        self.csv_path = csv_path
        self.csv_obj_desc_path = csv_obj_desc_path
        self.out_dir = out_dir
        self.gd_model = gd_model
        self.sam_predictor = sam_predictor
        self.device = device
        self.dataset = dataset.upper()

        # Thresholds (box and nms can be adapted)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold
        self.nms_threshold = nms_threshold
        self.ada_box_threshold = ada_box_threshold
        self.ada_nms_threshold = ada_nms_threshold
        self.nms_strategy = nms_strategy
        self.penalty_score = penalty_score
        self.max_images = max_images

        # Directories
        self.kept_dir = out_dir / "kept"
        self.discarded_dir = out_dir / "discarded"
        self.lbl_kept_path = out_dir / "kept.csv"
        self.lbl_discarded_path = out_dir / "discarded.csv"

        # Statistics
        self.kept_count = 0
        self.total_masks = 0
        self.coverage_list = []
        self.pollution_list = []
        self.missing_distractors_count = 0
        self.missing_dino_detections_count = 0

        # Constants
        self.MIN_AREA_THRESHOLD = 0.005
        self.MAX_AREA_THRESHOLD = 0.40

        # Object descriptions (concept -> description)
        if self.dataset == 'FLUX':
            self.concept2desc = self._load_concept2desc(csv_obj_desc_path)

    def _load_concept2desc(self, csv_path) -> dict:
        
        if csv_path is None:
            logger.warning("No csv path provided for object descriptions. Only concept names will be used as caption.")
            return {}

        desc_df = pd.read_csv(csv_path)
        concept2desc = {}
        for _, row in desc_df.iterrows():
            desc = row['description']
            desc = desc.split('with')[0].strip()
            concept2desc[row['concept']] = desc

        return concept2desc

    def create_directories(self):
        """
        Create necessary output directories.
        """
        os.makedirs(self.out_dir, exist_ok = True)
        os.makedirs(self.kept_dir, exist_ok = True)
        os.makedirs(self.discarded_dir, exist_ok = True)

    def compute_nms_threshold_distractors(self, num_distractors: int) -> float:
        """
        Compute appropriate NMS threshold, based on the number of distractors. If the image 
        is not very crowded, it is unlikely that there are duplicate boxes. NMS threshold 
        not aggressive. More aggressive if crowded image with an high number of distractors.

        Args:
            num_distractors (int): Number of distractors in the image.

        Returns:
            float: NMS threshold.
        """
        if num_distractors < 10:
            return 0.70
        elif num_distractors < 30:
            return 0.50
        else:
            return 0.30

    def compute_nms_threshold_logits(self, logits: torch.Tensor) -> float:
        """
        Compute appropriate NMS threshold based on the variance of logits. If high variance, 
        it is unlikely that there are duplicate boxes, so a higher NMS threshold is used.

        Args:
            logits (torch.Tensor): Logits tensor from the model.

        Returns:
            float: NMS threshold.
        """
        logits_np = logits.cpu().numpy()
        logits_variance = np.var(logits_np)

        logits_norm_variance = min(logits_variance, 1.0)
        nms_threshold = 0.30 + logits_norm_variance * 0.40  # [0.30, 0.70]

        return nms_threshold

    def compute_nms_threshold_boxes(self, boxes: torch.Tensor, height: int, width: int) -> float:
        """
        Compute appropriate NMS threshold based on the overlap of bounding boxes. If there is a high average overlap 
        between boxes, a lower NMS threshold is used to reduce redundancy. Otherwise, a higher NMS threshold is used.

        Args:
            boxes (torch.Tensor): Bounding boxes in cxcywh format. They must be converted in xyxy.
            height (int): Height of the image to process.
            width (int): Width of the image to process.

        Returns:
            float: NMS threshold.
        """
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([width, height, width, height])   # from cxcywh format to xyxy
        iou_matrix, _ = box_ops.box_iou(boxes_xyxy, boxes_xyxy)

        mask = ~torch.eye(len(boxes), dtype=torch.bool)
        avg_overlap = iou_matrix[mask].mean().item()

        nms_threshold = 0.70 - (avg_overlap * 0.40) # [0.30, 0.70]

        return nms_threshold

    def get_adaptive_nms_threshold(self, num_distractors: int, logits: torch.Tensor, boxes: torch.Tensor, height: int, width: int) -> float:
        """
        Get NMS threshold using the selected strategy. The strategy can be num_distractors, logits variance or boxes overlapping.
        """
        if not self.ada_nms_threshold or self.nms_threshold is not None:    # if NMS manually set
            return self.nms_threshold

        if self.nms_strategy == "num_distractors":
            return self.compute_nms_threshold_distractors(num_distractors)
        elif self.nms_strategy == "logits_variance":
            return self.compute_nms_threshold_logits(logits)
        elif self.nms_strategy == "boxes_overlap":
            return self.compute_nms_threshold_boxes(boxes, height, width)
        else:
            logger.warning(f"Error: unknown NMS strategy '{self.nms_strategy}'. Using default.")
            return 0.50

    def process_single_image(self, row: pd.Series, lbl_kept_writer: csv.DictWriter, lbl_discarded_writer: csv.DictWriter) -> bool:
        """
        Processes one single image using GSAM. Grounding DINO is used to predict bounding boxes, 
        SAM to generate masks. Generated masks are then compared against the ground truth. If a 
        GSAM mask with IoU greater than iou value specified as parameter against the Ground 
        Truth exists, the current image is kept. Otherwise it is temporarily discarded.

        Args:
            row (pd.Series): Row from CSV file containing image information.
            lbl_kept_writer (csv.DictWriter): CSV writer for saving kept samples information.
            lbl_discarded_writer (csv.DictWriter): CSV writer for saving discarded samples information.

        Returns:
            bool: True if the image was successfully processed. Otherwise False.
        """
        if self.dataset == 'O3':
            image_name = row['image_name']
            class_name = row['target_type']
            num_distractors = row['num_distractors']

            if self.ada_box_threshold:
                self.box_threshold = 0.25 if num_distractors >= 30 else 0.30

        elif self.dataset == 'FLUX':
            odd = row['odd_name']
            a = row['distractor_a_name']
            b = row['distractor_b_name']
            
            # search for corresponding scene
            pattern = f"{odd}_{a}_{b}_v*.jpg"
            matches = sorted(self.img_dir.glob(pattern))
            if not matches:
                logger.warning(f"No image found for pattern '{pattern}'")
                return False
            if len(matches) > 1:
                logger.warning(f"Multiple images found for pattern '{pattern}'. Using '{matches[0].name}'")

            image_name = matches[0].name
            num_distractors = 2    # not used because fixed NMS

            if self.csv_obj_desc_path:
                class_name = f"{self.concept2desc[odd]} . {self.concept2desc[a]} . {self.concept2desc[b]}"
            else:
                class_name = f"{odd} . {a} . {b}"
        
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        logger.info(f"\nProcessing '{image_name}' (caption '{class_name}')")

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
                text_threshold = self.text_threshold,
                nms_threshold = None,  # apply later. We need logits, boxes. Line 231
            )

            boxes, logits, phrases = keep_valid_boxes(boxes, logits, phrases, self.MIN_AREA_THRESHOLD, self.MAX_AREA_THRESHOLD)

            # 3. Check detections
            if boxes is None or len(boxes) == 0:
                logger.warning(f"No Grounding DINO detections for '{image_name}'")
                self.missing_dino_detections_count += 1
                return False

            # 4. Compute NMS threshold to apply
            nms_threshold = self.get_adaptive_nms_threshold(num_distractors, logits, boxes, height, width)
            if nms_threshold is not None:
                boxes, logits, phrases = apply_nms(boxes, logits, phrases, nms_threshold)

            # 5. Load binary ground truth mask for odd and distractors (only for O3)
            if self.dataset == 'O3':
                gt_odd_mask_bin = load_gt_mask(self.gt_odd_dir, image_name)
                if gt_odd_mask_bin is None:
                    logger.warning(f"Error during loading of odd ground truth mask for '{image_name}'")
                    return False
                
                gt_dist_mask_bin = load_gt_mask(self.gt_dist_dir, image_name)
                if gt_dist_mask_bin is None:
                    logger.warning(f"Error during loading of distractors ground truth mask for '{image_name}'")
                    return False

            # 6. Run segmentation model
            self.sam_predictor.set_image(image)

            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([width, height, width, height])   # from cxcywh format to xyxy
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image.shape[:2]).to(self.device)

            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )

            # 7. Masks evaluation and distractors metrics computation
            masks_info = []

            for i, mask_tensor in enumerate(masks):
                self.total_masks += 1
                mask_np = mask_tensor[0].detach().cpu().numpy()
                detected_phrase = phrases[i] if i < len(phrases) else class_name

                iou = 0.0
                is_the_odd = False

                if self.dataset == 'O3':
                    iou = compute_iou(mask_np, gt_odd_mask_bin)
                else:
                    # is_the_odd = detected_phrase.lower().strip().replace(' ', '') in self.concept2desc[odd].lower().replace(' ', '')
                    is_the_odd = detected_phrase.lower().strip().replace(' ', '').endswith(self.concept2desc[odd].lower().split()[-1])

                masks_info.append({
                    "index": i,
                    "box": boxes_xyxy[i].cpu().numpy(),
                    "mask": mask_np,
                    "iou": iou,
                    "is_odd": is_the_odd,
                    "phrase": detected_phrase,
                    "logit": logits[i].item() if i < len(logits) else 0.0
                })
            
            masks_info = apply_nms_on_masks(masks_info, iou_threshold = 0.2)
            if not masks_info:
                logger.warning(f'No valid masks for {image_name} after NMS')
                return False
            
            base_name = Path(image_name).stem

            if self.dataset == 'O3':
                best = max(masks_info, key = lambda x: x["iou"])    # best mask   
                
                coverage, pollution = 0, 0 
                distractors_masks = [info["mask"] for info in masks_info if info["index"] != best["index"]] # exclude ODD mask
                if distractors_masks:
                    distractors_mask_union = np.zeros_like(distractors_masks[0], dtype=bool)
                    for mask in distractors_masks:
                        distractors_mask_union = distractors_mask_union | mask.astype(bool)
                    coverage, pollution = compute_coverage_pollution_distractors(distractors_mask_union, gt_dist_mask_bin)
                else:
                    logger.warning(f"No distractors masks found for '{image_name}.")
                    self.missing_distractors_count += 1

                # 8 Save results
                is_kept = best["iou"] >= self.iou_threshold
                lbl_writer = lbl_kept_writer if is_kept else lbl_discarded_writer
                target_dir = self.kept_dir if is_kept else self.discarded_dir
                out_image_dir = target_dir / base_name
                os.makedirs(out_image_dir, exist_ok = True)

                # 8a. save original image and corresponding ground truth for reference
                Image.fromarray(to_numpy_image(image)).save(out_image_dir / f"{base_name}__img.png")
                Image.fromarray((gt_odd_mask_bin * 255).astype(np.uint8)).save(out_image_dir / f"{base_name}__gt.png")
            else:
                lbl_writer = lbl_kept_writer
                target_dir = self.kept_dir
                out_image_dir = target_dir / base_name
                os.makedirs(out_image_dir, exist_ok = True)

            for info in masks_info:
                if self.dataset == 'O3':
                    is_odd = (info["index"] == best["index"]) and is_kept
                else:
                    is_odd = info["is_odd"]

                # 8b. save masks odd and normal
                mask_suffix = "__ODD" if is_odd else ""
                mask_filename = f"{base_name}__mask_box{info['index']}{mask_suffix}.png"
                mask_path = out_image_dir / mask_filename
                Image.fromarray((info["mask"] * 255).astype(np.uint8)).save(mask_path)

                # 8c. write to appropriate CSV file
                lbl_writer.writerow({
                    "image_name": image_name,
                    "mask_filename": str(mask_path.relative_to(self.out_dir)),
                    "is_odd": int(is_odd),
                    "iou": f"{info['iou']:.3f}",
                    "confidence": f"{info['logit']:.3f}",
                    "target_type": class_name if self.dataset == 'O3' else class_name.split('.')[0].strip(),
                    "num_distractors": num_distractors
                })

            # 9. Log result
            if self.dataset == 'O3':
                if is_kept:
                    self.kept_count += 1
                    self.coverage_list.append(coverage)
                    self.pollution_list.append(pollution)
                    # print(f"{image_name}: coverage = {coverage:.3f} - pollution = {pollution:.3f}")
                    logger.info(f" KEPT - best IoU = {best['iou']:.3f}")
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
        csv_sep = ";" if self.dataset == "O3" else ","
        img_props = pd.read_csv(self.csv_path, sep = csv_sep)
        total_images = len(img_props) if self.max_images is None else min(len(img_props), self.max_images)
        logger.info(f"Processing {total_images} images from {len(img_props)} total.")

        # Process images
        with open(self.lbl_kept_path, mode = 'w', newline = '') as lbl_kept_file, \
            open(self.lbl_discarded_path, mode = 'w', newline = '') as lbl_discarded_file:

            lbl_kept_writer = csv.DictWriter(lbl_kept_file, fieldnames=[
                "image_name", "mask_filename", "is_odd", "iou", 
                "confidence", "target_type", "num_distractors"
            ])
            lbl_discarded_writer = csv.DictWriter(lbl_discarded_file, fieldnames=[
                "image_name", "mask_filename", "is_odd", "iou", 
                "confidence", "target_type", "num_distractors"
            ])
            lbl_kept_writer.writeheader()
            lbl_discarded_writer.writeheader()

            for _, row in tqdm(img_props.iloc[:total_images].iterrows(), 
                total=total_images, desc="GSAM Labeling Progress"):
                self.process_single_image(row, lbl_kept_writer, lbl_discarded_writer) 

        logger.info('\n========== GSAM LABELING FINISHED ==========')
        logger.info(f"Results of labeling saved in {self.out_dir}")  
        logger.info(f"Kept {self.kept_count} images out of {total_images}: {(self.kept_count/total_images * 100):.2f} %.") 

        # Average and std coverage and pollution
        avg_coverage, std_coverage = np.mean(self.coverage_list), np.std(self.coverage_list)
        avg_pollution, std_pollution = np.mean(self.pollution_list), np.std(self.pollution_list)

        return (self.kept_count, round(avg_coverage, 4), round(std_coverage, 4), round(avg_pollution, 4), 
                round(std_pollution, 4), self.missing_dino_detections_count, self.missing_distractors_count)

def main(args):

    logger.info('\n========== GSAM LABELING ==========')

    # Initialize and run labeler
    labeler = GSAMDatasetLabeler(
        root=args.root_dir,
        img_dir=args.img_dir,
        gt_odd_dir=args.gt_odd_dir,
        csv_path=args.csv_path,
        csv_obj_desc_path=args.csv_obj_desc_path,
        out_dir=args.out_dir,
        gd_model=args.gd_model,  
        sam_predictor=args.sam_predictor, 
        device=args.device,
        dataset=args.dataset,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        iou_threshold=args.iou_threshold,
        ada_box_threshold=args.ada_box_threshold,
        max_images=args.max_images,
    )

    labeler.run()

    logger.info('\n========== GSAM LABELING FINISHED ==========')
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
    parser.add_argument('--csv-obj-desc-path', type=Path, default=None,
        help='Path to object descriptions csv file (for FLUX generated scenes)')
    parser.add_argument('--gd-model', required=True)    # pass as path or str
    parser.add_argument('--sam-predictor', required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='O3')
    parser.add_argument('--box-threshold', type=float, default=0.30)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument('--iou-threshold', type=float, default=0.75)
    parser.add_argument('--ada-box-threshold', type=bool, default=False)
    parser.add_argument('--max-images', type=int, default=None, 
        help='Maximum number of images to process (for testing purposes)')

    args = parser.parse_args()

    args.device = torch.device(args.device)

    if args.img_dir is None:
        args.img_dir = args.root_dir / 'images'

    if args.gt_odd_dir is None:
        args.gt_odd_dir = args.root_dir / 'targ_labels'

    if args.csv_path is None:
        args.csv_path = args.root_dir / 'image_properties.csv'  # triplets_final_cross_category.csv