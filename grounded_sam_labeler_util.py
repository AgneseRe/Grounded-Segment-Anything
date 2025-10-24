import torch
import numpy as np

from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

def load_gt_mask(gt_dir: Path, image_name: str):
    """
    Find the corresponding ground truth mask based on the image name.

    Args:
        root (Path): The root directory containing data.
        image_name (str): The name of the image.

    Returns:
        The path to the corresponding ground truth mask if found. Otherwise None.
    """
    base_name = Path(image_name).stem  # without extension
    expected_gt_path = gt_dir / (base_name + ".jpg")
    if expected_gt_path.exists():
        return expected_gt_path
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


def apply_nms_on_masks(masks, scores, iou_threshold: float = 0.9) -> list[int]:

    scores = np.asarray(scores)

    sorted_indices = np.argsort(scores)[::-1]
    
    kept_indices = []
    
    while sorted_indices.size > 0:
        current_idx_in_original = sorted_indices[0]
        kept_indices.append(current_idx_in_original)
        
        # Rimuove l'indice corrente
        sorted_indices = sorted_indices[1:]
        
        if sorted_indices.size == 0:
            break
            
        current_mask = masks[current_idx_in_original]
        
        # Calcola IoU con le maschere rimanenti e filtra
        remaining_indices = []
        for next_idx_in_original in sorted_indices:
            iou = compute_iou(current_mask, masks[next_idx_in_original])
            if iou < iou_threshold:
                remaining_indices.append(next_idx_in_original)
        
        sorted_indices = np.array(remaining_indices)
        
    return kept_indices


def to_numpy_image(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1,3):
            img = np.transpose(img, (1,2,0))  # da CHW a HWC
    if img.dtype != np.uint8:
        img = (img * 255).clip(0,255).astype(np.uint8)
    return img


# ========== BOUNDING BOX REFINEMENT ==========
# Group Evidence Matters: Tiling-based Semantic Gating for Dense Object Detection
def spatial_gate_dbscan(boxes: torch.Tensor, logits: torch.Tensor, width: int, height: int, 
    eps: float, min_samples: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    boxes_np = boxes.cpu().numpy()
    logits_np = logits.cpu().numpy()

    # obtain centroids, the first two columns of cxcywh boxes and denormalize
    centroids = boxes_np[:, :2]
    denormalized_centroids = np.copy(centroids)
    denormalized_centroids[:, 0] *= width
    denormalized_centroids[:, 1] *= height

    # apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(denormalized_centroids)
    labels = clustering.labels_

    clustering_indices = np.where(labels != -1)
    if len(clustering_indices) == 0:
        return np.array(), np.array(), np.array()

    clustering_labels = labels[clustering_indices]
    clustering_logits = logits_np[clustering_indices]
    clustering_boxes = boxes_np[clustering_indices]

    return clustering_boxes, clustering_logits, clustering_labels

def semantic_gate_dbscan(embeddings: np.ndarray, spatial_labels: np.ndarray, 
    eps: float, min_samples: int = 2) -> np.ndarray:
    
    final_labels = np.full(len(embeddings), -1, dtype=int)
    unique_spatial_labels = np.unique(spatial_labels)
    cluster_counter = 0

    for label in unique_spatial_labels:
        if label == -1: # isolated boxes
            isolated_indices = np.where(spatial_labels == label)
            final_labels[isolated_indices] = -1
            continue

        # current cluster
        cluster_indices = np.where(spatial_labels == label)
        cluster_embeddings = embeddings[cluster_indices]
        
        if len(cluster_embeddings) < min_samples:
            continue    # too much small

        # apply DBSCAN
        clustering_semantic = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(cluster_embeddings)
        
        semantic_labels = clustering_semantic.labels_
        
        valid_semantic_indices_in_cluster = np.where(semantic_labels != -1)
        
        if len(valid_semantic_indices_in_cluster) > 0:
            global_indices_of_validated = cluster_indices[valid_semantic_indices_in_cluster]
            final_labels[global_indices_of_validated] = cluster_counter
            cluster_counter += 1

    return final_labels

def weighted_average_box(boxes: np.ndarray, logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    merged_boxes = []
    merged_logits = []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        group_indices = np.where(labels == label)
        group_boxes = boxes[group_indices]
        group_logits = logits[group_indices]

        if label == -1: # isolated
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
    final_boxes_xyxy = np.concatenate(merged_boxes, axis=0) if merged_boxes else np.array()
    final_scores = np.concatenate(merged_logits, axis=0) if merged_logits else np.array()

    return final_boxes_xyxy, final_scores
  
