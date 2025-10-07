import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import json
import csv
import os
from typing import Optional, Dict, List, Tuple
import logging

# Grounding DINO
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GSAMDatasetLabeler:
    
    def __init__(
        self,
        csv_path: Path,
        img_dir: Path,
        root_dir: Path,
        out_root: Path,
        gd_model,
        sam_predictor,
        device,
        compute_iou,
        load_gt_mask,
        to_numpy_image,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        max_images: Optional[int] = None
    ):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.root_dir = root_dir
        self.out_root = out_root
        self.gd_model = gd_model
        self.sam_predictor = sam_predictor
        self.device = device
        
        # Thresholds
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold
        self.max_images = max_images

        self.compute_iou = compute_iou
        self.load_gt_mask = load_gt_mask
        self.to_numpy_image = to_numpy_image
        
        # Directories
        self.kept_dir = out_root / "kept"
        self.discarded_dir = out_root / "discarded"
        self.lbl_path = out_root / "labels.csv"
        
        # COCO structure
        self.coco = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.annotation_id = 1
        self.category_map = {}  # category_name -> category_id
        
    def setup_directories(self):
        """Crea le directory necessarie"""
        os.makedirs(self.kept_dir, exist_ok=True)
        os.makedirs(self.discarded_dir, exist_ok=True)
        os.makedirs(self.out_root, exist_ok=True)
        
    def get_or_create_category(self, category_name: str) -> int:
        """Ottiene o crea una categoria COCO"""
        if category_name not in self.category_map:
            cat_id = len(self.category_map) + 1
            self.category_map[category_name] = cat_id
            self.coco["categories"].append({
                "id": cat_id,
                "name": category_name,
                "supercategory": "object"
            })
        return self.category_map[category_name]
    
    def mask_to_coco_rle(self, mask: np.ndarray) -> Dict:
        """Converti maschera binaria in formato RLE COCO (semplificato)"""
        # Nota: questa è una versione semplificata. Per COCO reale usa pycocotools
        from itertools import groupby
        pixels = mask.flatten()
        rle = []
        for value, group in groupby(pixels):
            rle.append(len(list(group)))
        return {"counts": rle, "size": list(mask.shape)}
    
    def process_single_image(
        self,
        img_id: int,
        row: pd.Series,
        lbl_writer: csv.DictWriter
    ) -> bool:
        """
        Processa una singola immagine
        Returns: True se processata con successo, False altrimenti
        """
        image_name = row['image_name']
        class_name = row['target_type']
        
        logger.info(f"\nProcessing '{image_name}' (class '{class_name}')")
        
        try:
            # 1. Carica immagine
            image_path = self.img_dir / image_name
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                return False
            
            image, image_transformed = load_image(str(image_path))
            height, width, _ = image.shape
            
            # 2. Run Grounding DINO
            boxes, logits, phrases = predict(
                model=self.gd_model,
                image=image_transformed,
                caption=class_name,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )
            
            # 3. Carica ground truth
            gt_path = self.load_gt_mask(self.root_dir, image_name)
            if gt_path is None:
                logger.warning(f"Ground truth not found for '{image_name}'")
                return False
            
            gt_mask = Image.open(gt_path).convert(mode="L")
            gt_mask_np = np.array(gt_mask)
            gt_mask_bin = (gt_mask_np > 127).astype(np.uint8)
            
            # 4. Registra immagine COCO
            self.coco["images"].append({
                "id": img_id,
                "file_name": image_name,
                "width": width,
                "height": height
            })
            
            # 5. Check detections
            if boxes is None or len(boxes) == 0:
                logger.warning(f"No detections for '{image_name}'")
                return False
            
            # 6. Run SAM
            self.sam_predictor.set_image(image)
            
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([width, height, width, height])
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
                boxes_xyxy, image.shape[:2]
            ).to(self.device)
            
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            
            # 7. Valuta maschere
            masks_info = []
            
            for i, mask_tensor in enumerate(masks):
                mask_np = mask_tensor[0].detach().cpu().numpy()
                iou = self.compute_iou(mask_np, gt_mask_bin)
                
                masks_info.append({
                    "index": i,
                    "box": boxes_xyxy[i].cpu().numpy(),
                    "mask": mask_np,
                    "iou": iou,
                    "phrase": phrases[i] if i < len(phrases) else class_name,
                    "logit": logits[i].item() if i < len(logits) else 0.0
                })
            
            # 8. Trova la migliore
            best = max(masks_info, key=lambda x: x["iou"])
            base_name = Path(image_name).stem
            
            # 9. Salva risultati
            is_kept = best["iou"] >= self.iou_threshold
            target_dir = self.kept_dir if is_kept else self.discarded_dir
            out_image_dir = target_dir / base_name
            os.makedirs(out_image_dir, exist_ok=True)
            
            # Salva immagine originale e GT
            Image.fromarray(self.to_numpy_image(image)).save(
                out_image_dir / f"{base_name}__img.png"
            )
            Image.fromarray((gt_mask_bin * 255).astype(np.uint8)).save(
                out_image_dir / f"{base_name}__gt.png"
            )
            
            # Salva maschere e crea annotations
            category_id = self.get_or_create_category(class_name)
            
            for info in masks_info:
                is_odd = (info["index"] == best["index"]) and is_kept
                
                # Salva maschera
                mask_suffix = "__ODD" if is_odd else ""
                mask_filename = f"{base_name}__mask_box{info['index']}{mask_suffix}.png"
                mask_path = out_image_dir / mask_filename
                Image.fromarray((info["mask"] * 255).astype(np.uint8)).save(mask_path)
                
                # Scrivi nel CSV
                lbl_writer.writerow({
                    "image_name": image_name,
                    "mask_filename": str(mask_path.relative_to(self.out_root)),
                    "is_odd": int(is_odd),
                    "is_kept": int(is_kept),
                    "iou": f"{info['iou']:.3f}",
                    "confidence": f"{info['logit']:.3f}",
                    "category": info["phrase"]
                })
                
                # Aggiungi annotation COCO (solo per kept)
                if is_kept:
                    # Calcola bounding box dalla maschera
                    y_indices, x_indices = np.where(info["mask"])
                    if len(y_indices) > 0:
                        x_min, x_max = x_indices.min(), x_indices.max()
                        y_min, y_max = y_indices.min(), y_indices.max()
                        bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                        area = int(info["mask"].sum())
                        
                        self.coco["annotations"].append({
                            "id": self.annotation_id,
                            "image_id": img_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "area": area,
                            "segmentation": self.mask_to_coco_rle(info["mask"]),
                            "iscrowd": 0,
                            "iou_with_gt": float(info["iou"]),
                            "is_best": is_odd
                        })
                        self.annotation_id += 1
            
            # 10. Log risultato
            if is_kept:
                logger.info(f"✓ KEPT - best IoU = {best['iou']:.3f} (mask {best['index']} marked as ODD)")
            else:
                logger.info(f"✗ DISCARDED - best IoU = {best['iou']:.3f} < {self.iou_threshold}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing '{image_name}': {str(e)}")
            return False
    
    def run(self):
        """Esegue il labeling completo"""
        logger.info("="*80)
        logger.info("Starting GSAM Dataset Labeling")
        logger.info("="*80)
        
        # Setup
        self.setup_directories()
        
        # Carica CSV
        img_props = pd.read_csv(self.csv_path, sep=",")
        total_images = len(img_props) if self.max_images is None else min(len(img_props), self.max_images)
        logger.info(f"Processing {total_images} images from {len(img_props)} total")
        
        # Apri file CSV per labels
        lbl_file = open(self.lbl_path, 'w', newline='')
        lbl_writer = csv.DictWriter(lbl_file, fieldnames=[
            "image_name", "mask_filename", "is_odd", "is_kept", 
            "iou", "confidence", "category"
        ])
        lbl_writer.writeheader()
        
        # Processa immagini
        img_id = 1
        try:
            for index, row in img_props.iterrows():
                if self.max_images is not None and index >= self.max_images:
                    break
                
                if self.process_single_image(img_id, row, lbl_writer):
                    img_id += 1
                    
        finally:
            lbl_file.close()
        
        # Salva COCO
        coco_path = self.out_root / "gsam_annotations.json"
        with open(coco_path, "w") as f:
            json.dump(self.coco, f, indent=2)