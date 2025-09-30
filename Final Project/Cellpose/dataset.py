"""
Segmentation Dataset for Cellpose training pipeline with support for CopyPaste augmentation and Albumentations.
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from cellpose.dynamics import masks_to_flows_gpu
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from skimage.measure import label, regionprops
import fastremap

from copy_paste import CopyPaste


class SegmentationDataset(Dataset):
    """
    A custom PyTorch Dataset designed for cell instance segmentation using Cellpose-style supervision.

    This dataset handles image loading, instance mask processing, advanced data augmentation with Albumentations,
    CopyPaste-based object blending, and automatic computation of flow maps and cell probabilities.
    """

    def __init__(self, images, masks, file_ids, is_train,
                 image_size=(256, 256), device=None):
        """
        Initialize the dataset.

        Args:
            images (list): List of input image arrays (HWC or CHW format).
            masks (list): List of corresponding instance masks, either (H, W) or (N, H, W).
            file_ids (list): List of string identifiers for each sample.
            is_train (bool): Whether the dataset is used for training (enables augmentation).
            image_size (tuple): Desired output resolution (height, width).
            device (torch.device or str or None): Device to compute Cellpose flow ("cuda", "mps", "cpu").
        """
        self.images = images
        self.masks = masks
        self.file_ids = file_ids
        self.image_size = image_size

        if device is None:
            self.device = (
                torch.device('cuda') if torch.cuda.is_available()
                else torch.device('mps') if torch.backends.mps.is_available()
                else torch.device('cpu')
            )
        else:
            self.device = torch.device(device)
            if self.device.type == 'cuda' and not torch.cuda.is_available():
                print("Warning: CUDA not available. Falling back to CPU.")
                self.device = torch.device('cpu')
            if self.device.type == 'mps' and not torch.backends.mps.is_available():
                print("Warning: MPS not available. Falling back to CPU.")
                self.device = torch.device('cpu')

        self.is_train = is_train
        self.copy_paste_transform_instance = (
            CopyPaste(keep_prob=0.5, select_prob=0.9, occ_thresh=0.7)
            if is_train else None
        )

        if is_train:
            self.train_augmentation_transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(
                    scale=(0.7, 1.3),
                    translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                    rotate=(-180, 180),
                    shear=0.0,
                    interpolation=cv2.INTER_LINEAR,
                    p=0.5,
                ),
                A.OneOf([
                    A.ElasticTransform(alpha=50, sigma=2.5, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                ], p=0.2),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.25),
                A.RandomCrop(height=image_size[0], width=image_size[1], p=1.0),
            ], is_check_shapes=False)
        else:
            self.train_augmentation_transforms = None

        self.final_transforms = A.Compose([
            A.ToFloat(max_value=255.0),
            ToTensorV2()
        ], is_check_shapes=False)

    def __len__(self):
        """
        return the number of samples in the dataset.
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.images)

    def _load_raw_data(self, idx):
        """
        載入原始影像和掩碼，並確保它們的格式符合後續處理的要求。

        Args:
            idx (int): 資料集中的索引。

        Returns:
            tuple: (image_np, mask_np_inst, file_id)
                - image_np (np.ndarray): (H, W, C) 格式的影像 NumPy 陣列 (C=3)。
                - mask_np_inst (np.ndarray): (N, H, W) 格式的布林實例掩碼 NumPy 陣列。
                - file_id (str): 影像的檔案 ID。
        """
        image_np = self.images[idx]
        mask_np = self.masks[idx] # mask_np 從這裡讀取時應該是 (H, W) uint16

        file_id = self.file_ids[idx]

        # 確保 image_np 是 (H, W, C) 格式
        image_np = np.array(image_np)
        if image_np.ndim == 3 and image_np.shape[0] < image_np.shape[-1]: # 檢查是否為 (C, H, W)
            image_np = image_np.transpose(1, 2, 0) # 轉換為 (H, W, C)
        elif image_np.ndim == 2: # 灰度圖轉換為 RGB (假彩色，3 通道)
            image_np = np.stack([image_np, image_np, image_np], axis=-1)
        
        # 確保 mask_np 是 (N, H, W) 格式 (布林類型)
        if not isinstance(mask_np, np.ndarray):
            raise TypeError(f"Raw mask_np for {file_id} is not a NumPy array. Type: {type(mask_np)}.")
        
        # 處理 (H, W) 格式的實例 ID 遮罩 (這應該是你 load_images_and_masks 返回的格式)
        if mask_np.ndim == 2:
            unique_ids = np.unique(mask_np)
            # 排除背景 0
            instance_masks = [(mask_np == uid) for uid in unique_ids if uid != 0]
            if len(instance_masks) > 0:
                mask_np_inst = np.stack(instance_masks, axis=0) # (N, H, W) 布林
            else:
                mask_np_inst = np.zeros((0, *image_np.shape[:2]), dtype=bool) # 空的 (0, H, W)
        elif mask_np.ndim == 3 and mask_np.shape[0] >= 1: # 假設它已經是 (N, H, W) 或 (1, H, W) 布林
            mask_np_inst = mask_np.astype(bool) # 確保是布林類型
        else:
            raise ValueError(f"Raw mask_np for {file_id} has unexpected dimensions: {mask_np.shape}. Expected 2D (H,W) or 3D (N,H,W).")

        # 再次檢查確保 mask_np_inst 是布林類型
        mask_np_inst = mask_np_inst.astype(bool)

        return image_np, mask_np_inst, file_id

    def __getitem__(self, idx):
        """
        get a single sample from the dataset.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            dict: A dictionary containing the following keys:
                - 'image': Transformed image tensor (C, H, W).
                - 'flows': Flow tensor (2, H, W).
                - 'cellprobs': Cell probability tensor (H, W).
                - 'masks': Instance mask tensor (N, H, W) with unique IDs.
                - 'file_id': Identifier for the sample.
        """
        # 載入當前影像和掩碼 (作為主要影像 Image1)
        image1_np, mask1_np_inst, file_id1 = self._load_raw_data(idx)

        # --- 新增的檢查：如果原始掩碼為空，則直接返回 None ---
        # 檢查掩碼是否包含任何有效像素 (即非零像素)
        # 或者檢查第一個維度 (實例數) 是否為 0
        if mask1_np_inst.shape[0] == 0 or np.sum(mask1_np_inst) == 0:
            # print(f"DEBUG: Original mask for {file_id1} is empty or contains no valid pixels. Skipping this sample.")
            return None

        # 初始化當前處理的圖片和掩碼，默認為原始數據
        current_image_np = image1_np
        current_mask_np_inst = mask1_np_inst

        # --- 應用 CopyPaste (僅在訓練模式下) ---
        if self.is_train and self.copy_paste_transform_instance is not None:
            # 隨機選擇第二張圖片的索引，確保不選擇到同一張
            if random.random() < 0.2:
                idx2 = idx
                while idx2 == idx:
                    idx2 = random.randint(0, len(self) - 1)

                # 載入第二張影像和掩碼 (作為貼上來源 Image2)
                image2_np, mask2_np_inst, file_id2 = self._load_raw_data(idx2)

                # 檢查第二張圖片的掩碼是否為空，如果為空則不能用於 CopyPaste
                if mask2_np_inst.shape[0] == 0 or np.sum(mask2_np_inst) == 0:
                    # print(f"WARNING: Source mask for CopyPaste ({file_id2}) is empty. Skipping CopyPaste for this pair, using original data for {file_id1}.")
                    # 如果第二張圖片是空的，則不執行 CopyPaste，使用原始數據
                    pass  # current_image_np 和 current_mask_np_inst 保持為原始值
                else:
                    # 將 NumPy 陣列轉換為 PyTorch 張量，準備給 CopyPaste
                    # CopyPaste 預期 (C, H, W) 的浮點張量，值範圍 [0, 1]
                    image1_tensor_cp = torch.from_numpy(
                        image1_np).permute(2, 0, 1).float() / 255.0
                    mask1_tensor_cp = torch.from_numpy(
                        mask1_np_inst)  # (N, H, W) bool
                    label1_tensor_cp = torch.ones(
                        mask1_tensor_cp.shape[0], dtype=torch.long)
                    # 這裡調用 get_ltrb_from_mask 會在 copy_paste.py 中被修正為處理空掩碼
                    ltrb1_tensor_cp = CopyPaste.get_ltrb_from_mask(
                        mask1_tensor_cp)

                    image2_tensor_cp = torch.from_numpy(
                        image2_np).permute(2, 0, 1).float() / 255.0
                    mask2_tensor_cp = torch.from_numpy(
                        mask2_np_inst)  # (N, H, W) bool
                    label2_tensor_cp = torch.ones(
                        mask2_tensor_cp.shape[0], dtype=torch.long)
                    ltrb2_tensor_cp = CopyPaste.get_ltrb_from_mask(
                        mask2_tensor_cp)

                    # 構造一個包含兩張圖片的 batch，用於 CopyPaste 的 apply 方法
                    image_batch_cp = torch.stack(
                        [image1_tensor_cp, image2_tensor_cp], dim=0)  # (2, C, H, W)
                    annots_cp = {
                        "masks": [mask1_tensor_cp, mask2_tensor_cp],
                        "labels": [label1_tensor_cp, label2_tensor_cp],
                        "ltrbs": [ltrb1_tensor_cp, ltrb2_tensor_cp]
                    }

                    try:
                        # 執行 CopyPaste 轉換：將 image_batch_cp[1] 的物件貼到
                        # image_batch_cp[0] 上
                        cp_image_tensor, cp_mask_tensor, _, _ = \
                            self.copy_paste_transform_instance.apply(
                                image=image_batch_cp,
                                annots=annots_cp,
                                idx1=0,  # 指定第一張圖片作為目標
                                idx2=1  # 指定第二張圖片作為來源
                            )

                        # 檢查 CopyPaste 輸出是否有效
                        if cp_image_tensor is not None and cp_mask_tensor.numel() > 0:
                            # 將 Copy-Paste 的結果轉換回 NumPy (H, W, C) 和 (N, H, W)
                            # 格式
                            current_image_np = (
                                cp_image_tensor.permute(
                                    1,
                                    2,
                                    0).cpu().numpy() *
                                255.0).astype(
                                np.uint8)
                            current_mask_np_inst = cp_mask_tensor.cpu().numpy().astype(bool)
                        else:
                            # CopyPaste 失敗或沒有有效物件，使用原始圖片和掩碼
                            # (current_image_np/mask 已是原始數據)
                            print(
                                f"WARNING: CopyPaste produced empty result for {file_id1} and {file_id2}. Using original data.")

                    except Exception as e:
                        # CopyPaste 執行時發生錯誤，回退到原始圖片和掩碼
                        print(
                            f"WARNING: CopyPaste failed for {file_id1} and {file_id2}: {e}. Using original data.")

        # --- 應用 Albumentations 增強 (作用於 Copy-Paste 結果或原始圖片) ---
        # 1. 首先應用訓練專屬的幾何和色彩轉換 (只在訓練模式下)
        if self.is_train and self.train_augmentation_transforms is not None:
            # Albumentations 期望 masks 是一個列表，且內部 transform (如 Rotate)
            # 要求 mask 的 dtype 為 uint8，因此在傳入前進行轉換
            augmented_data = self.train_augmentation_transforms(
                image=current_image_np,
                masks=[m.astype(np.uint8) for m in current_mask_np_inst]
            )
            current_image_np = augmented_data['image']

            # Albumentations 處理完後，將 masks 重新堆疊並轉回 bool 類型
            if len(augmented_data['masks']) > 0:
                current_mask_np_inst = np.stack(
                    augmented_data['masks']).astype(bool)
            else:
                current_mask_np_inst = np.zeros(
                    (0, *current_image_np.shape[:2]), dtype=bool)

        # 2. 最後應用共同轉換 (包含 Resize, ToFloat, ToTensorV2)
        # 這裡 ToTensorV2 會將 NumPy 陣列轉換為 PyTorch 張量
        final_transformed_data = self.final_transforms(
            image=current_image_np,
            masks=[m.astype(np.uint8) for m in current_mask_np_inst]
        )
        # (C, H, W) PyTorch 張量
        image_transformed = final_transformed_data['image']
        # 列表 of PyTorch 張量 (可能為 (1, H, W) 或 (H, W))
        mask_transformed_list = final_transformed_data['masks']

        # 將 mask_transformed_list 中的每個 mask 堆疊成 (N, H, W) 布林張量
        if len(mask_transformed_list) > 0:
            # 確保每個 mask 在堆疊前都是 2D (H, W)
            mask_transformed = torch.stack(
                [m.squeeze(0) if m.ndim == 3 else m for m in mask_transformed_list]).bool()
        else:
            # 如果沒有物件，創建一個空的 (0, H, W) 布林張量
            mask_transformed = torch.zeros(
                (0, *image_transformed.shape[1:]), dtype=torch.bool)

        # 將 (N, H, W) 的 PyTorch 實例掩碼轉換為 Cellpose 期望的單個 2D 掩碼 (H, W)，值為 uint16
        reindexed_mask_np = torch.zeros(
            mask_transformed.shape[1:], dtype=torch.long)  # 使用 torch.long
        if mask_transformed.numel() > 0:
            for i, instance_mask in enumerate(mask_transformed):
                reindexed_mask_np[instance_mask] = i + 1  # 為每個實例賦予從 1 開始的唯一 ID
        reindexed_mask_np = reindexed_mask_np.cpu(
        ).numpy().astype(np.uint16)  # 轉換為 NumPy uint16

        # --- 檢查和修復掩碼中的連通組件問題 ---
        if np.sum(reindexed_mask_np > 0) == 0:
            # print(f"DEBUG: Mask {file_id1} empty after augmentations. Returning None.")
            return None

        unique_labels = np.unique(reindexed_mask_np)
        if len(unique_labels) > 1 and 0 in unique_labels:  # 排除背景 (0)
            processed_mask_np = np.zeros_like(
                reindexed_mask_np, dtype=np.uint16)
            new_label_id = 1  # 從 1 開始分配新的 ID

            for label_val in unique_labels:
                if label_val == 0:
                    continue  # 跳過背景
                instance_mask = (reindexed_mask_np == label_val)
                labeled_components = label(
                    instance_mask, connectivity=2)  # 找出當前實例中的所有連通組件
                props = regionprops(labeled_components)

                if len(props) > 1:  # 如果有多個組件，只保留最大的
                    largest_component_area = 0
                    largest_component_label_in_props = 0
                    for p in props:
                        if p.area > largest_component_area:
                            largest_component_area = p.area
                            largest_component_label_in_props = p.label
                    processed_mask_np[labeled_components ==
                                      largest_component_label_in_props] = new_label_id
                    new_label_id += 1
                elif len(props) == 1:  # 如果只有一個組件，直接使用
                    processed_mask_np[instance_mask] = new_label_id
                    new_label_id += 1
            reindexed_mask_np = processed_mask_np

        # 再次檢查處理後是否仍有有效掩碼
        if np.sum(reindexed_mask_np > 0) == 0:
            # print(f"DEBUG: Mask {file_id1} empty after component processing. Returning None.")
            return None

        # 使用 fastremap 重新編號掩碼 ID，確保是連續的
        if reindexed_mask_np.max() > 0:
            reindexed_mask_np = fastremap.renumber(
                reindexed_mask_np, in_place=True)[0]

        # --- 計算 flows 和 cellprobs ---
        # 初始為全零陣列
        # Cellpose flow (2, H, W)
        flow_np = np.zeros((2, *self.image_size), dtype=np.float32)
        cellprob_np = np.zeros(self.image_size,
                               dtype=np.float32)  # Cell probability (H, W)

        try:
            if self.is_train:
                # 在訓練模式下，如果 flow 計算失敗，則返回 None (跳過此樣本)
                if reindexed_mask_np.shape != self.image_size:
                    # 這應該不會發生，因為 Albumentations 的 RandomCrop 應該確保了尺寸
                    print(
                        f"Warning: Mask shape {reindexed_mask_np.shape} does not match image_size {self.image_size} before flow calculation for {file_id1}. This indicates a potential issue with RandomCrop or CopyPaste output size.")

                vecn = masks_to_flows_gpu(
                    np.ascontiguousarray(reindexed_mask_np),
                    device=self.device)
                vecn = vecn.astype(np.float32)
                if vecn.ndim != 3 or vecn.shape[0] != 2:
                    raise ValueError(
                        f"masks_to_flows_gpu returned unexpected shape: {vecn.shape} for {file_id1}.")
                flow_np = vecn
                cellprob_np = (reindexed_mask_np > 0).astype(np.float32)
            else:
                # 在驗證模式下，如果 flow 計算失敗，則靜默處理 (flow 和 cellprob 保持為零)
                try:
                    vecn = masks_to_flows_gpu(
                        np.ascontiguousarray(reindexed_mask_np), device=self.device)
                    vecn = vecn.astype(np.float32)
                    if vecn.ndim == 3 and vecn.shape[0] == 2:
                        flow_np = vecn
                        cellprob_np = (
                            reindexed_mask_np > 0).astype(
                            np.float32)
                except Exception as e:
                    pass  # 計算失敗，flow_np 和 cellprob_np 保持為零

        except Exception as e:
            if self.is_train:
                print(
                    f"ERROR: Skipping Mask {file_id1} due to critical error in flow calculation (train): {e}.")
                return None  # 訓練模式下，如果發生嚴重錯誤，跳過樣本
            else:
                pass  # 驗證模式下忽略錯誤

        # --- 轉換為 PyTorch 張量並返回 ---
        flow_tensor = torch.from_numpy(flow_np)
        cellprob_tensor = torch.from_numpy(cellprob_np)
        reindexed_mask_tensor = torch.from_numpy(
            reindexed_mask_np)  # 2D uint16 mask

        return {
            'image': image_transformed,
            'flows': flow_tensor,
            'cellprobs': cellprob_tensor,
            'masks': reindexed_mask_tensor,
            'file_id': file_id1
        }
