"""
# Cellpose Training Script
This script is designed to train, validate, or test a Cellpose model for image segmentation tasks.
"""

from cellpose import plot as cp_plot
from dataset import SegmentationDataset
import re
from cellpose import metrics as cp_metrics
from utils import (
    get_image_patches, get_mask_patches
)
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch import nn
import os
import time
import gc

import numpy as np
np.float = float


# 確保這個導入可以找到你的 Cellpose plot 模組


def extract_id(filename):
    """
    extracts the file ID from a filename.
    Args:
        filename (str): The filename from which to extract the ID.
                        Expected format: "{id}_img.npy" or "{id}_mask.npy".
    Returns:
        str: The extracted file ID.
    """
    return re.sub(r'_(img|mask)\.npy$', '', filename)


def _loss_fn_seg(lbl, y, device):
    """
    Cellpose segmentation loss function.
    Args:
        lbl (torch.Tensor): Ground truth labels, shape (B, 4, H, W).
                            Channel 0: Instance labels,
                            Channel 1: Cell probability target,
                            Channel 2: Flow Y,
                            Channel 3: Flow X.
        y (torch.Tensor): Model predictions, shape (B, 4, H, W).
                          Channel 0: Predicted instance labels,
                          Channel 1: Predicted cell probability logits,
                          Channel 2: Predicted flow Y,
                          Channel 3: Predicted flow X.
        device (torch.device): Device to perform calculations on.
    Returns:
        tuple: Total loss, flow loss, cell probability loss.
               - total_loss (float): Combined loss value.
               - flow_loss (float): Loss for flow predictions.
               - cellprob_loss (float): Loss for cell probability predictions.
    """
    criterion_mse = nn.MSELoss(reduction="mean")
    criterion_bce = nn.BCEWithLogitsLoss(reduction="mean")

    # === Flow Loss ===
    true_flows = 5. * lbl[:, -2:]            # (B, 2, H, W)
    pred_flows = y[:, -3:-1]                 # (B, 2, H, W)
    flow_loss = criterion_mse(pred_flows, true_flows) / 2.0

    # === Cell Probability Loss ===
    true_cellprob = (lbl[:, -3] > 0.5).float()  # (B, H, W)
    pred_cellprob_logits = y[:, -1]            # (B, H, W)
    cellprob_loss = criterion_bce(pred_cellprob_logits, true_cellprob)

    total_loss = flow_loss + cellprob_loss

    return total_loss, flow_loss.item(), cellprob_loss.item()


def mask_to_rgb_cv2(mask_np):
    """
    map a 2D mask of instance IDs to a BGR color mask.
    Args:
        mask_np (np.ndarray): 2D array of instance IDs, shape (H, W).
    Returns:
        np.ndarray: BGR color mask, shape (H, W, 3).
    """
    if mask_np.ndim != 2:
        raise ValueError("Mask must be a 2D array.")

    unique_ids = np.unique(mask_np)
    unique_ids = unique_ids[unique_ids != 0]  # 排除背景ID

    h, w = mask_np.shape
    bgr_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # 為每個實例ID分配一個隨機顏色
    # 使用一個可重現的種子以便每次訓練可視化時顏色一致
    rng = np.random.default_rng(42)
    colors = {}
    for uid in unique_ids:
        # OpenCV 使用 BGR 格式
        colors[uid] = rng.integers(0, 256, 3, dtype=np.uint8)

    for uid in unique_ids:
        bgr_mask[mask_np == uid] = colors[uid]
    return bgr_mask


def tensor_to_cv2_image(img_tensor_chw):
    """
    Convert a PyTorch tensor in CHW format to a normalized OpenCV image.
    Args:
        img_tensor_chw (torch.Tensor): Input image tensor in CHW format.
                                       Expected shape: (C, H, W) where C=1 or C=3.
    Returns:
        np.ndarray: Normalized OpenCV image in HWC format.
                    If input is grayscale (C=1), output will be (H, W).
                    If input is RGB (C=3), output will be (H, W, 3).
    """
    img_np_chw = img_tensor_chw.cpu().numpy()

    min_val = img_np_chw.min()
    max_val = img_np_chw.max()

    if max_val == min_val:
        img_np_norm = np.full_like(img_np_chw, 0.5)
    else:
        img_np_norm = (img_np_chw - min_val) / (max_val - min_val)

    # 轉換為 0-255 並轉為 uint8
    img_np_uint8 = (img_np_norm * 255).astype(np.uint8)

    if img_np_uint8.shape[0] == 1:  # 灰度圖 (1, H, W)
        return img_np_uint8.squeeze(0)  # 移除通道維度，變成 (H, W)
    elif img_np_uint8.shape[0] == 3:  # 彩色圖 (3, H, W)
        return np.transpose(img_np_uint8, (1, 2, 0))  # CHW -> HWC
    else:
        raise ValueError(
            f"Unsupported image tensor shape: {img_tensor_chw.shape}")

# --- 原有的 collate_fn 函數 ---


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Args:
        batch (list): List of samples, each sample is a dict with keys:
                      'image', 'flows', 'cellprobs', 'masks', 'file_id'.
    Returns:
        dict: A dictionary with batched tensors for 'image', 'flows', 'cellprobs', 'masks', and 'file_id'.
    """
    # 過濾掉 __getitem__ 返回的 None 樣本
    # 這裡的 item is not None 是關鍵
    batch = [item for item in batch if item is not None]

    if not batch:  # 如果批次為空，例如所有樣本都失敗了
        # 在這裡可以選擇：
        # 1. 返回 None，然後在訓練循環中跳過這個批次
        # 2. 拋出一個錯誤，如果批次為空是個嚴重的問題
        # 這裡我們選擇返回 None，與訓練循環中的處理方式一致
        print("WARNING: Empty batch after filtering out invalid samples. Returning None.")
        return None

    images = [item['image'] for item in batch]
    flows = [item['flows'] for item in batch]
    cellprobs = [item['cellprobs'] for item in batch]
    masks = [item['masks'] for item in batch]
    file_ids = [item['file_id'] for item in batch]

    images_batch = torch.stack(images, 0)
    flows_batch = torch.stack(flows, 0)
    cellprobs_batch = torch.stack(cellprobs, 0)
    masks_batch = torch.stack(masks, 0)

    return {
        'image': images_batch,
        'flows': flows_batch,
        'cellprobs': cellprobs_batch,
        'masks': masks_batch,
        'file_id': file_ids
    }

# --- 原有的 save_npy_raw_only 函數 ---


def save_npy_raw_only(data_dir):
    """
    Convert Sartorius dataset images and masks to .npy format.
    Args:
        data_dir (str): Directory to save the .npy files.
                        Expected to contain images in './sartorius/train' and annotations in './sartorius/train.csv'.
    """

    image_dir = './sartorius/train'
    csv_path = './sartorius/train.csv'

    df = pd.read_csv(csv_path)

    all_file_ids = df['id'].unique()

    for file_id in tqdm(all_file_ids, desc="Converting to .npy"):
        image_path = os.path.join(image_dir, f"{file_id}.png")
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found for {file_id}. Skipping.")
            continue
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        height, width = image_np.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint16)

        image_annotations = df[df['id'] == file_id].copy()
        image_annotations['annotation'] = image_annotations['annotation'].astype(
            str)

        current_instance_idx = 1
        for rle_str in image_annotations['annotation']:
            try:
                rle_parts = list(map(int, rle_str.split()))
                if len(rle_parts) % 2 != 0:
                    continue

                temp_mask_segment = np.zeros(
                    (height * width,), dtype=np.uint16)
                is_valid_rle_segment = True
                for i_rle in range(0, len(rle_parts), 2):
                    start_pixel = rle_parts[i_rle] - 1
                    length = rle_parts[i_rle + 1]
                    end_pixel = start_pixel + length
                    if start_pixel < 0 or end_pixel > height * width:
                        is_valid_rle_segment = False
                        break
                    temp_mask_segment[start_pixel:end_pixel] = current_instance_idx

                if not is_valid_rle_segment:
                    continue

                mask.flat[temp_mask_segment >
                          0] = temp_mask_segment[temp_mask_segment > 0]
                current_instance_idx += 1

            except ValueError:
                pass
            except Exception:
                pass
        np.save(os.path.join(data_dir, f"{file_id}_img.npy"), image_np)
        np.save(os.path.join(data_dir, f"{file_id}_mask.npy"), mask)

# --- 原有的 load_images_and_masks 函數 ---


def load_images_and_masks(data_dir, ids, split_name):
    """
    Load images and masks from .npy files.
    Args:
        data_dir (str): Directory containing the .npy files.
        ids (list): List of file IDs to load.
        split_name (str): Name of the dataset split (e.g., "train", "validation").
    Returns:
        tuple: A tuple containing:
            - images (list): List of loaded images as numpy arrays.
            - masks (list): List of loaded masks as numpy arrays.
            - file_ids_list (list): List of file IDs corresponding to the loaded images and masks.
    """

    images, masks, file_ids_list = [], [], []

    # 確定檔案都是 .npy 格式
    img_extension = '.npy'
    mask_extension = '.npy'

    for id in tqdm(ids, desc=f"Loading {split_name} images"):
        img_path = os.path.join(data_dir, f"{id}_img{img_extension}")
        mask_path = os.path.join(data_dir, f"{id}_mask{mask_extension}")

        if not os.path.exists(img_path):
            print(f"Warning: Image file not found at {img_path}. Skipping.")
            continue
        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found at {mask_path}. Skipping.")
            continue

        try:
            # *** 使用 np.load 讀取 .npy 檔案 ***
            img = np.load(img_path)
            msk = np.load(mask_path)

            # 檢查圖片和掩碼的空間形狀是否一致
            if img.shape[:2] != msk.shape[:2]:
                print(
                    f"Warning: Image shape {img.shape[:2]} and mask shape {msk.shape[:2]} mismatch for {id}. Skipping.")
                continue

            # 確保掩碼是 uint16 類型，因為 Cellpose 期望實例 ID
            if msk.dtype != np.uint16:
                msk = msk.astype(np.uint16)

            images.append(img)
            masks.append(msk)
            file_ids_list.append(id)

        except Exception as e:
            print(f"Error loading {id}: {e}. Skipping.")
    return images, masks, file_ids_list


# --- 原有的 load_data 函數 ---
def load_data(train_data_dir, valid_data_dir, args):
    """
    Load training and validation data from specified directories.
    Args:
        train_data_dir (str): Directory containing training data .npy files.
        valid_data_dir (str): Directory containing validation data .npy files.
        args (argparse.Namespace): Command line arguments containing batch size and image size.
    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for training data.
            - val_loader (DataLoader): DataLoader for validation data.
    """

    # 確保訓練數據目錄存在
    if not os.path.exists(train_data_dir) or not any(
            f.endswith('_img.npy') for f in os.listdir(train_data_dir)):
        raise ValueError(f"Training data directory '{train_data_dir}' not found or contains no .npy image files. "
                         "Please ensure your training data is prepared.")

    # 確保驗證數據目錄存在
    if not os.path.exists(valid_data_dir) or not any(
            f.endswith('_img.npy') for f in os.listdir(valid_data_dir)):
        raise ValueError(f"Validation data directory '{valid_data_dir}' not found or contains no .npy image files. "
                         "Please ensure your validation data is prepared.")

    # 從各自的目錄加載 ID
    train_ids = sorted(set(extract_id(f) for f in os.listdir(
        train_data_dir) if f.endswith('_img.npy')))
    val_ids = sorted(set(extract_id(f) for f in os.listdir(
        valid_data_dir) if f.endswith('_img.npy')))

    if not train_ids:
        raise ValueError(
            f"No .npy files found in training folder: {train_data_dir}.")
    if not val_ids:
        raise ValueError(
            f"No .npy files found in validation folder: {valid_data_dir}.")

    # 載入圖像和掩碼
    full_train_images, full_train_masks, train_original_file_ids = load_images_and_masks(
        train_data_dir, train_ids, "train")
    val_images, val_masks, val_file_ids = load_images_and_masks(
        valid_data_dir, val_ids, "validation")

    print(f"Total images loaded: {len(train_ids) + len(val_ids)}")
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")

    if not full_train_images:
        raise ValueError("No training images loaded.")
    if not val_images:
        raise ValueError("No validation images loaded.")

    patch_height, patch_width = 301, 302
    stride_height, stride_width = 73, 134

    processed_train_images, processed_train_masks, processed_train_file_ids = [], [], []
    for img, mask, original_file_id in zip(
            full_train_images, full_train_masks, train_original_file_ids):
        img_patches = get_image_patches(
            img, (patch_height, patch_width), (stride_height, stride_width))
        mask_patches = get_mask_patches(
            mask, (patch_height, patch_width), (stride_height, stride_width))
        processed_train_images.extend(img_patches)
        processed_train_masks.extend(mask_patches)
        processed_train_file_ids.extend(
            [f"{original_file_id}_patch_{j}" for j in range(len(img_patches))])

    train_dataset = SegmentationDataset(
        processed_train_images,
        processed_train_masks,
        processed_train_file_ids,
        image_size=(args.image_size, args.image_size),
        is_train=True
    )

    val_dataset = SegmentationDataset(
        val_images,
        val_masks,
        val_file_ids,
        image_size=(args.image_size, args.image_size),
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return train_loader, val_loader


class PlottingTools:  # 模擬 cp_plot
    """
    A simplified version of Cellpose's plotting tools for visualizing flow and cell probability maps.
    This class provides methods to convert flow vectors to circular visualizations.
    """

    def dx_to_circ(self, dxy):
        """
        Convert flow vectors to a circular visualization.
        Args:
            dxy (np.ndarray): Flow vectors, expected shape (2, H, W) or (3, H, W) for 3D.
                              The first channel is Y-flow, the second channel is X-flow.
        Returns:
            np.ndarray: RGB image representing the flow vectors.
        """

        # 簡易模擬 Cellpose 的 dx_to_circ 視覺化，實際可能更複雜
        # dxy 預期是 (2, H, W) 或 (3, H, W) for 3D
        # 這是一個簡化版本，只用於基本的 RGB 映射
        flow_mag = np.linalg.norm(dxy, axis=0)  # (H, W)
        flow_angle = np.arctan2(dxy[0], dxy[1])  # (H, W), Y-flow, X-flow

        # 將角度映射到色相 (0-180 for OpenCV H channel)
        h = (flow_angle + np.pi) / (2 * np.pi) * 179  # 0-179
        s = np.ones_like(flow_mag) * 255  # 飽和度
        v = np.clip(flow_mag / (flow_mag.max() + 1e-8)
                    * 255, 0, 255)  # 亮度，歸一化到 0-255

        hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb


cp_plot = PlottingTools()  # 實例化以便調用

# --- Trainer 類別修改部分 ---


class Trainer:
    """
    Trainer class for managing the training and validation of a Cellpose model.
    This class handles the training loop, validation, and model saving.
    Attributes:
        device (torch.device): Device to run the model on (CPU or GPU).
        model (CellposeModel): The Cellpose model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        args (argparse.Namespace): Command line arguments containing training parameters.
        class_weights (torch.Tensor, optional): Class weights for handling class imbalance.
        train_losses (list): List to store training losses for each epoch.
        maps (list): List to store validation mean average precision (mAP) for each epoch.
        best_map (float): Best validation mAP achieved during training.
        mask_losses (list): List to store mask losses for each epoch.
        flow_losses (list): List to store flow losses for each epoch.
        output_viz_dir (str): Directory to save validation prediction visualizations.
    """

    def __init__(self, device, model, optimizer, args, class_weights=None):
        """
        Initializes the Trainer with the model, optimizer, and training parameters.
        Args:
            device (torch.device): Device to run the model on (CPU or GPU).
            model (CellposeModel): The Cellpose model to train.
            optimizer (torch.optim.Optimizer): Optimizer for training the model.
            args (argparse.Namespace): Command line arguments containing training parameters.
            class_weights (torch.Tensor, optional): Class weights for handling class imbalance.
        """

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.train_losses = []
        self.maps = []
        self.best_map = 0.0
        self.mask_losses = []
        self.flow_losses = []
        self.class_weights = class_weights

        self.output_viz_dir = os.path.join(
            args.saved_path, "validation_predictions")
        os.makedirs(self.output_viz_dir, exist_ok=True)

        self.eval_cellprob_threshold = getattr(args, 'cellprob_threshold', 0.0)
        self.eval_flow_threshold = getattr(args, 'flow_threshold', 0.4)
        self.eval_min_size = getattr(args, 'min_size', 15)
        self.eval_bsize = args.batch_size

        self.learning_rate = args.learning_rate
        self.n_epochs = args.epochs
        self._setup_lr_schedule()

    def _setup_lr_schedule(self):
        """
        Setup the learning rate schedule based on the number of epochs.
        This method calculates a learning rate schedule that starts from a small value
        and gradually increases to the specified learning rate, then stabilizes or decreases
        based on the number of epochs.
        """

        # Cellpose 原始的學習率調度邏輯
        LR = np.linspace(1e-6, self.learning_rate, 10)
        LR = np.append(LR, self.learning_rate *
                       np.ones(max(0, self.n_epochs - 10)))
        if self.n_epochs > 300:
            LR = LR[:-100]
            for i in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(10))
        elif self.n_epochs > 99:
            LR = LR[:-50]
            for i in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(5))
        self.lr_schedule = LR  # 將計算出的學習率列表保存為實例變數

    def train(self, train_loader, epoch):
        """
        Train the model for one epoch.
        Args:
            train_loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.
        Returns:
            float: Average loss for the epoch.
        """

        # 【關鍵修改】: 在每個 Epoch 開始時，手動設置學習率
        if epoch < len(self.lr_schedule):  # 確保 epoch 索引在 LR_schedule 範圍內
            current_lr = self.lr_schedule[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            # 打印確認
            print(
                f"\n--- Epoch {epoch + 1}/{self.n_epochs}, Current Learning Rate: {current_lr:.6f} ---")
        else:
            # 如果 epoch 超出了 lr_schedule 的長度，表示 LR 已經達到穩定值
            # 或者你希望在此之後保持最後一個 LR
            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"\n--- Epoch {epoch + 1}/{self.n_epochs}, Learning Rate (unchanged): {current_lr:.6f} ---")

        self.model.net.train()
        total_loss = 0.0

        train_output_vis_dir = os.path.join(
            self.args.saved_path, f"train_vis_epoch_{epoch+1}")
        os.makedirs(train_output_vis_dir, exist_ok=True)

        debug_print_frequency = 1 if epoch < 2 else 100

        for batch_idx, batch in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)):
            if batch is None:
                # 這個警告應該在 `load_data` 或 DataLoader 級別處理，
                # 但在這裡再次確認跳過空批次是必要的。
                # 最好在數據加載時就篩選掉無效樣本，而不是在訓練循環中跳過。
                print(
                    f"Skipping empty batch at Epoch {epoch+1}, Batch {batch_idx}.")
                continue

            # Ensure all tensors are moved to device AND are float32
            images_tensor = batch['image'].to(self.device)
            # 確保傳入 `float()` 以避免潛在的 Dtype 錯誤
            flows_tensor = batch['flows'].to(self.device).float()
            cellprobs_tensor = batch['cellprobs'].to(self.device).float()
            # Masks are instance IDs, should be float for loss calculation
            masks_tensor = batch['masks'].to(self.device).float()

            # Construct `lbl` target tensor for Cellpose loss function
            lbl_combined = torch.cat([
                masks_tensor.unsqueeze(1),       # Channel 0: Instance labels
                cellprobs_tensor.unsqueeze(1),
                # Channel 1: Cell probability target
                # Channels 2, 3: Flow targets (Y, X)
                flows_tensor
            ], dim=1)  # Final shape: [B, 4, H, W]

            self.optimizer.zero_grad()
            outputs_raw = self.model.net(images_tensor)
            # This is `y` in _loss_fn_seg, should be float32 from model
            model_output = outputs_raw[0]

            # Calculate segmentation loss (flow and cell probability)
            # 確保 _loss_fn_seg 能夠處理你傳入的 tensor 類型和維度
            loss_seg, loss_flow_val, loss_cellprob_val = _loss_fn_seg(
                lbl_combined, model_output, self.device)

            # Ensure loss_seg is 0-d tensor (as discussed for previous error)
            # 這段處理 `loss_seg.ndim` 的邏輯是正確的
            if loss_seg.ndim > 0:
                loss = loss_seg.squeeze()
            else:
                loss = loss_seg

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # --- 可視化和保存中間結果 (只在每個 Epoch 的第一個 Batch 進行) ---
            if batch_idx == 0:
                print(
                    f"  Saving training visualization for Epoch {epoch+1} Batch {batch_idx}...")
                idx_to_vis = 0

                # 確保這裡的索引與 model_output 的通道順序一致
                # flowsY, flowsX
                vis_flow_preds = model_output[idx_to_vis, 0:2, :, :]
                # cellprob logits
                vis_cellprob_preds = model_output[idx_to_vis, 2, :, :]

                # 這些輔助函數 (tensor_to_cv2_image, mask_to_rgb_cv2, cp_plot) 需要被定義或正確導入
                # 否則這裡會報錯 NameError
                try:
                    # --- 原始圖像 ---
                    original_img_np = tensor_to_cv2_image(
                        images_tensor[idx_to_vis])
                    if original_img_np.ndim == 2:
                        original_img_np = cv2.cvtColor(
                            original_img_np, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(
                        os.path.join(
                            train_output_vis_dir,
                            f"epoch_{epoch+1}_batch_{batch_idx}_original_img.png"),
                        original_img_np)

                    # --- 真實 Mask ---
                    true_mask_np = batch['masks'][idx_to_vis].cpu().numpy()
                    true_mask_rgb = mask_to_rgb_cv2(true_mask_np)
                    cv2.imwrite(
                        os.path.join(
                            train_output_vis_dir,
                            f"epoch_{epoch+1}_batch_{batch_idx}_true_mask.png"),
                        true_mask_rgb)

                    # --- 真實 Flow ---
                    true_flow_np_for_plot = batch['flows'][idx_to_vis].cpu(
                    ).numpy()
                    true_flow_rgb = cp_plot.dx_to_circ(true_flow_np_for_plot)
                    true_flow_bgr = cv2.cvtColor(
                        true_flow_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        os.path.join(
                            train_output_vis_dir,
                            f"epoch_{epoch+1}_batch_{batch_idx}_true_flow.png"),
                        true_flow_bgr)

                    # --- 預測 Cell Probability Map ---
                    pred_cellprob_np = torch.sigmoid(
                        vis_cellprob_preds).detach().cpu().numpy()
                    pred_cellprob_img = (
                        pred_cellprob_np *
                        255).astype(
                        np.uint8)
                    cv2.imwrite(
                        os.path.join(
                            train_output_vis_dir,
                            f"epoch_{epoch+1}_batch_{batch_idx}_pred_cellprob.png"),
                        pred_cellprob_img)

                    # --- 預測 Flow ---
                    pred_flow_np_for_plot = vis_flow_preds.detach().cpu().numpy()
                    pred_flow_rgb = cp_plot.dx_to_circ(pred_flow_np_for_plot)
                    pred_flow_bgr = cv2.cvtColor(
                        pred_flow_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        os.path.join(
                            train_output_vis_dir,
                            f"epoch_{epoch+1}_batch_{batch_idx}_pred_flow.png"),
                        pred_flow_bgr)
                except NameError as e:
                    print(
                        f"WARNING: Visualization skipped due to missing imports/definitions: {e}")
                    print(
                        "Please ensure `_loss_fn_seg`, `tensor_to_cv2_image`, `mask_to_rgb_cv2`, `cp_plot` are correctly defined or imported.")

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, valid_loader, epoch):
        """
        Validate the model on the validation dataset.
        Args:
            valid_loader (DataLoader): DataLoader for validation data.
            epoch (int): Current epoch number.
        Returns:
            tuple: A tuple containing:
                - None (placeholder for future metrics, if needed).
                - float: Mean average precision (mAP) score for the validation set.
        """

        self.model.net.eval()

        all_gt_masks = []
        all_pred_masks = []

        viz_count = 0
        max_viz_per_epoch = 3

        for batch_idx, batch in enumerate(
                tqdm(valid_loader, desc=f"Epoch {epoch+1} [Validate]", leave=False)):
            if batch is None:
                print(
                    f"Skipping empty validation batch at Epoch {epoch}, Batch {batch_idx}.")
                continue

            images_for_cellpose = []
            original_images_np = []
            gt_masks_np = batch['masks'].cpu().numpy()

            for img_tensor in batch['image']:
                img_np = img_tensor.cpu().numpy()
                if img_np.shape[0] == 3 or img_np.shape[0] == 1:
                    img_np = np.transpose(img_np, (1, 2, 0))
                images_for_cellpose.append(img_np)
                original_images_np.append(img_np)

            masks_pred_list, flows_pred_list, _ = self.model.eval(
                images_for_cellpose)

            for i in range(len(gt_masks_np)):
                gt_mask = gt_masks_np[i]
                pred_mask = masks_pred_list[i]

                all_gt_masks.append(gt_mask)
                all_pred_masks.append(pred_mask)

                # 可視化
                if viz_count < max_viz_per_epoch:
                    display_image = original_images_np[i]
                    if display_image.ndim == 3 and display_image.shape[-1] == 1:
                        display_image = display_image.squeeze(-1)
                    display_image_normalized = display_image.astype(np.float32)
                    if display_image_normalized.max() > 1.0 or display_image_normalized.min() < 0.0:
                        display_image_normalized = (display_image_normalized - display_image_normalized.min()) / (
                            display_image_normalized.max() - display_image_normalized.min())

                    gt_boundary_image = mark_boundaries(
                        display_image_normalized, gt_mask)
                    dt_boundary_image = mark_boundaries(
                        display_image_normalized, pred_mask)

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    axes[0].imshow(display_image_normalized)
                    axes[0].set_title("Original")
                    axes[1].imshow(gt_boundary_image)
                    axes[1].set_title("Ground Truth")
                    axes[2].imshow(dt_boundary_image)
                    axes[2].set_title("Prediction")
                    for ax in axes:
                        ax.axis('off')
                    plt.tight_layout()

                    viz_filename = os.path.join(
                        self.output_viz_dir,
                        f"epoch_{epoch+1}_compare_{batch['file_id'][i]}.png")
                    plt.savefig(viz_filename, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    viz_count += 1

        # === 使用 Cellpose 的 average_precision ===
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        ap, tp, fp, fn = cp_metrics.average_precision(
            all_gt_masks, all_pred_masks, threshold=thresholds)
        final_score = 0.0
        mean_ap = ap.mean(axis=0)
        log_file = os.path.join(self.args.saved_path, "map_log.txt")
        with open(log_file, "a", encoding="utf-8") as file:
            file.write(f"\n[Epoch {epoch+1}] Cellpose Metrics Evaluation\n")
            for i, th in enumerate(thresholds):
                file.write(
                    f"  AP@{th:.2f} = {mean_ap[i]:.4f} | TP={tp[:, i].sum()}, FP={fp[:, i].sum()}, FN={fn[:, i].sum()}\n")
            final_score = mean_ap.mean()
            file.write(
                f'>>> final score (mean average precision over all images) = {final_score:.4f}')

        print(f"[Epoch {epoch+1}] AP: {mean_ap}")
        return None, float(final_score)  # 返回 AP@0.5 作為指標

    def save_model(self, epoch, is_Best=False):
        """
        Save the model state and optimizer state to a checkpoint file.
        Args:
            epoch (int): Current epoch number.
            is_Best (bool): Whether this is the best model so far.
        """

        self.model.net.save_model(
            os.path.join(
                self.args.saved_path,
                f'best_model_{epoch}.pth'))

        if is_Best:
            self.model.net.save_model(
                os.path.join(
                    self.args.saved_path,
                    f'best_model_{self.best_map}.pth'))

    def load_model(self):
        """
        Load the model state and optimizer state from the latest checkpoint file.
        Returns:
            int: The epoch number to resume training from. Returns 0 if no checkpoint is found.
        """

        path = os.path.join(self.args.saved_path, 'latest_checkpoint.pth')
        if os.path.exists(path):
            print(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=self.device)
            self.model.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_map = checkpoint['best_map']
            self.train_losses = checkpoint['train_losses']
            self.maps = checkpoint['maps']
            self.mask_losses = checkpoint['mask_losses']
            self.flow_losses = checkpoint['flow_losses']
            print(
                f"Checkpoint loaded. Resuming from epoch {checkpoint['epoch'] + 1}")
            return checkpoint['epoch'] + 1
        return 0

# --- 原有的 train_model 函數 ---


def train_model(device, model, optimizer, train_loader, valid_loader, args):
    """
    Train the Cellpose model using the provided training and validation data loaders.
    Args:
        device (torch.device): Device to run the model on (CPU or GPU).
        model (CellposeModel): The Cellpose model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        args (argparse.Namespace): Command line arguments containing training parameters.
    """

    trainer = Trainer(device, model, optimizer, args)  # 不再需要 GT
    start_epoch = trainer.load_model()
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        loss = trainer.train(train_loader, epoch)
        trainer.train_losses.append(loss)
        print(f"Epoch {epoch + 1} training loss: {loss:.4f}")

        _, map50 = trainer.validate(valid_loader, epoch)

        if (map50 >= trainer.best_map):
            trainer.best_map = map50
            print(f"New best mAP50: {trainer.best_map:.4f}, saving model...")
            trainer.save_model(epoch, is_Best=True)
        else:
            trainer.save_model(epoch, is_Best=False)

        gc.collect()
        torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1} time: {time.time() - start:.2f} sec")


def validate_model(device, model, optimizer, valid_loader, args):
    """
    Validate the Cellpose model using the provided validation data loader.
    Args:
        device (torch.device): Device to run the model on (CPU or GPU).
        model (CellposeModel): The Cellpose model to validate.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        valid_loader (DataLoader): DataLoader for validation data.
        args (argparse.Namespace): Command line arguments containing validation parameters.
    """

    trainer = Trainer(device, model, optimizer, args)
    print("\n--- Starting Validation ---")
    _, map50 = trainer.validate(valid_loader, epoch=1000)
    print("--- Validation Complete ---")
    print(
        f"Validation mAP50: {map50:.4f}" if map50 is not None else "Validation mAP50 could not be computed.")
    gc.collect()
    torch.cuda.empty_cache()
