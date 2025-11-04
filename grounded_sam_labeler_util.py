import torch
import numpy as np

from PIL import Image
from pathlib import Path
from typing import List, Tuple
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from GroundingDINO.groundingdino.util import box_ops

def load_gt_mask(gt_dir: Path, image_name: str):
    """
    Find the ground truth mask based on the image name and convert it in binary format.

    Args:
        root (Path): The root directory containing data.
        image_name (str): The name of the image.

    Returns:
        The ground truth mask in binary format. Otherwise None.
    """
    base_name = Path(image_name).stem  # without extension
    expected_gt_path = gt_dir / (base_name + ".jpg")
    if expected_gt_path.exists():
        gt_mask = Image.open(expected_gt_path).convert(mode = "L")
        gt_mask_np = np.array(gt_mask)
        gt_mask_bin = (gt_mask_np > 127).astype(np.uint8)
        return gt_mask_bin
    else:
        return None
    

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
  """
  Compute the Intersection over Union between two binary masks.
  The masks are of the same size and they contains only 0 and 1.

  Args:
    pred_mask (np.ndarray): The predicted binary mask.
    gt_mask (np.ndarray): The ground truth binary mask.

  Returns:
    The Intersection over Union score between the two masks.
  """
  # print(f"pred_mask.shape = {pred_mask.shape}, gt_mask.shape = {gt_mask.shape}")
  # Explicitly convert masks as boolean values
  pred_mask_bool = pred_mask.astype(bool)
  gt_mask_bool = gt_mask.astype(bool)

  # Compute intersection and union area
  intersection_area = np.sum(np.logical_and(pred_mask_bool, gt_mask_bool))
  union_area = np.sum(np.logical_or(pred_mask_bool, gt_mask_bool))

  if union_area == 0: # if both masks empty, no ooo found
    return 0.0

  return float(intersection_area) / float(union_area)

def to_numpy_image(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1,3):
            img = np.transpose(img, (1,2,0))  # da CHW a HWC
    if img.dtype != np.uint8:
        img = (img * 255).clip(0,255).astype(np.uint8)
    return img


# ========== BOUNDING BOX REFINEMENT ==========

def keep_valid_boxes(boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], 
    min_area_threshold: float = 0.005, max_area_threshold: float = 0.40) -> Tuple[
        torch.Tensor, torch.Tensor, List[str]]:
    """
    Filters Grounding DINO bounding boxes, logits and phrases, maintaining only those 
    whose normalized area falls in an interval [min_area_threshold, max_area_threshold].

    Args:
        boxes (torch.Tensor): Tensor of bounding boxes in cxcywh format. Normalized to [0, 1].
        logits (torch.Tensor): Tensor of confidence scores for each bounding box.
        phrases (List[str]): List of phrases associated with each bounding box.
        min_area_threshold (float): Minimum area threshold for valid boxes.
        max_area_threshold (float): Maximum area threshold for valid boxes.

    Returns:
        tuple: Filtered boxes, logits, and phrases.
    """
    areas = boxes[:, 2] * boxes[:, 3]  # calculate areas as width * height

    valid_mask = (areas >= min_area_threshold) & (areas <= max_area_threshold)
    filtered_boxes = boxes[valid_mask]
    filtered_logits = logits[valid_mask]
    filtered_phrases = np.array(phrases)[valid_mask].tolist() # !!!

    return filtered_boxes, filtered_logits, filtered_phrases

# Group Evidence Matters: Tiling-based Semantic Gating for Dense Object Detection
# Yilun Xiao, https://www.arxiv.org/abs/2509.10779
def spatial_gate_dbscan(boxes: torch.Tensor, logits: torch.Tensor, width: int, height: int, 
    min_samples: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies DBSCAN clustering algorithm to spatially group bounding boxes close together.
    Two bounding boxes are close if the distance between their centroids is less than a 
    certain threshold, calculated based on the average diagonal length of all candidates.

    Args:
        boxes (torch.Tensor): Tensor of bounding boxes in cxcywh format. Normalized to [0, 1].
        logits (torch.Tensor): Tensor of confidence scores for each bounding box.
        width (int): Original width of the image in pixel.
        height (int): Original height of the image in pixel.
        min_samples (int, optional): Minimum number of samples to form a cluster. Default 2.

    Returns:
        tuple: original bounding boxes, original confindence scores and cluster labels.
    """
    boxes_np = boxes.cpu().numpy()
    logits_np = logits.cpu().numpy()

    centroids = boxes_np[:, :2]
    denormalized_centroids = np.copy(centroids)
    denormalized_centroids[:, 0] *= width
    denormalized_centroids[:, 1] *= height

    # Compute eps parameter for DBSCAN
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([width, height, width, height])
    boxes_xyxy_np = boxes_xyxy.cpu().numpy()
    x1, y1, x2, y2 = boxes_xyxy_np[:, 0], boxes_xyxy_np[:, 1], boxes_xyxy_np[:, 2], boxes_xyxy_np[:, 3]
    box_width = x2 - x1
    box_height = y2 - y1
    diagonals = np.sqrt(box_width**2 + box_height**2)
    average_diagonal = np.mean(diagonals)
    eps_spatial = 0.2 * average_diagonal    

    # Apply DBSCAN clustering algorithm
    clustering = DBSCAN(eps=eps_spatial, min_samples=min_samples).fit(denormalized_centroids)
    labels = clustering.labels_

    return boxes_np, logits_np, labels

def semantic_gate_dbscan(embeddings: np.ndarray, spatial_labels: np.ndarray, 
    eps: float, min_samples: int = 2) -> np.ndarray:
    """
    Verifies semantic coerence in each spatial cluster, applying DBSCAN on visual embeddings.
    """
    final_labels = np.full(len(embeddings), -1, dtype=int)
    unique_spatial_labels = np.unique(spatial_labels)
    cluster_counter = 0

    for label in unique_spatial_labels:
        if label == -1: # isolated boxes
            isolated_indices = np.where(spatial_labels == label)
            final_labels[isolated_indices] = -1
            continue

        cluster_indices = np.where(spatial_labels == label)
        cluster_embeddings = embeddings[cluster_indices]
        
        if len(cluster_embeddings) < min_samples:
            continue  

        clustering_semantic = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(cluster_embeddings)
        semantic_labels = clustering_semantic.labels_
        
        valid_semantic_indices_in_cluster = np.where(semantic_labels != -1)
        
        if len(valid_semantic_indices_in_cluster[0]) > 0:
            global_indices_of_validated = cluster_indices[0][valid_semantic_indices_in_cluster]
            final_labels[global_indices_of_validated] = cluster_counter
            cluster_counter += 1

    return final_labels

def weighted_average_box(boxes: np.ndarray, logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    merged_boxes = []
    merged_logits = []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        group_indices = np.where(labels == label)
        group_boxes = boxes[group_indices[0]]
        group_logits = logits[group_indices[0]]

        if label == -1:
            merged_boxes.append(group_boxes)
            merged_logits.append(group_logits)
        else:
            total_logit = np.sum(group_logits)
            weights = group_logits / total_logit
            merged_box = np.sum(group_boxes * weights[:, np.newaxis], axis=0)
            merged_logit = np.max(group_logits)

            merged_boxes.append(merged_box[np.newaxis, :])
            merged_logits.append(np.array([merged_logit])) 

    # final results combined
    final_boxes_cxcywh = np.concatenate(merged_boxes, axis=0) if merged_boxes else np.array()
    final_scores = np.concatenate(merged_logits, axis=0) if merged_logits else np.array()

    return final_boxes_cxcywh, final_scores
  
