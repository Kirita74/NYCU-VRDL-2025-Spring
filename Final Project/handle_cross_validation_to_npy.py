import json
import numpy as np
import os
from PIL import Image
from pycocotools import mask as mask_utils

# --- 1. 配置路徑 ---
json_file_path = 'crossvalidationfold5/coco_cell_valid_fold1.json'
image_root_path = 'sartorius'
output_dir = 'Cellpose_5fold_valid/fold1'

# --- 2. 建立輸出目錄 ---
os.makedirs(output_dir, exist_ok=True)
print(f"📁 輸出目錄已建立/存在: {output_dir}")

try:
    # --- 3. 載入 COCO JSON 檔案 ---
    print(f"📖 正在載入 COCO JSON 檔案: {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    print("✅ COCO JSON 載入成功。")

    images_info = coco_data['images']
    annotations_info = coco_data['annotations']

    image_id_to_annotations = {}
    for ann_data in annotations_info:
        image_id = ann_data['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann_data)

    print(f"🔍 總共找到 {len(images_info)} 張圖像。")

    # --- 4. 處理每張圖像並儲存為 .npy ---
    for i, img_info in enumerate(images_info):
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_width, img_height = img_info['width'], img_info['height']
        base_filename = os.path.splitext(os.path.basename(img_filename))[0]
        full_image_path = os.path.join(image_root_path, img_filename)

        print(f"[{i+1}/{len(images_info)}] 📷 處理圖像: {full_image_path}")

        # --- 5.1 載入圖像並轉為 RGB ---
        try:
            image = Image.open(full_image_path).convert('L')
            image_array = np.array(image)

            if image_array.dtype == np.uint16:
                image_array = image_array.astype(np.float32) / 65535.0
            elif image_array.dtype == np.uint8:
                image_array = image_array.astype(np.float32) / 255.0

            image_array = np.stack([image_array] * 3, axis=-1)  # (H, W, 3)

        except Exception as e:
            print(f"❌ 圖像載入失敗: {e}，已跳過")
            continue

        # --- 5.2 建立實例 segmentation mask ---
        annotations = image_id_to_annotations.get(img_id, [])
        mask_array = np.zeros((img_height, img_width), dtype=np.uint16)

        for idx, ann in enumerate(annotations):
            segmentation = ann.get('segmentation')

            if isinstance(segmentation, dict) and 'counts' in segmentation and 'size' in segmentation:
                try:
                    rle = segmentation
                    if isinstance(rle['counts'], list):
                        rle = mask_utils.frPyObjects([rle], rle['size'][0], rle['size'][1])[0]
                    binary_mask = mask_utils.decode(rle)
                    mask_array[binary_mask == 1] = idx + 1
                except Exception as e:
                    print(f"❌ RLE decode 錯誤 (ann_id={ann.get('id')}): {e}")
            else:
                print(f"⚠️ 非 RLE 格式 segmentation，ann_id={ann.get('id')}，已跳過")

        # --- 5.3 儲存 .npy ---
        img_out_path = os.path.join(output_dir, f"{base_filename}_img.npy")
        mask_out_path = os.path.join(output_dir, f"{base_filename}_mask.npy")

        np.save(img_out_path, image_array)
        np.save(mask_out_path, mask_array)

    print(f"\n✅ 所有圖像與遮罩已成功儲存至: {output_dir}")

except Exception as e:
    import traceback
    print(f"❌ 發生未知錯誤: {type(e).__name__}: {e}")
    traceback.print_exc()