import torch
import numpy as np

from pathlib import Path

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
  
