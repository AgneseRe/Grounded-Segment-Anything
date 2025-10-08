import torch
import numpy as np

from pathlib import Path

def load_gt_mask(gtdir: Path, image_name: str):
    """
    Find the corresponding ground truth mask based on the image name.

    Args:
        root (Path): The root directory containing data.
        image_name (str): The name of the image.

    Returns:
        The path to the corresponding ground truth mask if found. Otherwise None.
    """
    base_name = Path(image_name).stem  # without extension
    expected_gt_path = gtdir / (base_name + ".jpg")
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


def to_numpy_image(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1,3):
            img = np.transpose(img, (1,2,0))  # da CHW a HWC
    if img.dtype != np.uint8:
        img = (img * 255).clip(0,255).astype(np.uint8)
    return img
  
