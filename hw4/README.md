# NYCU Computer Vision 2025 Spring HW3
- StudentID: 313553037
- Name: 黃瑜明
## Introduction
Utilize Mask R-CNN, with ResNext-50 as the backbone to detect individual cell within an image. The model is trained to localize each cell by predicting its bounding box and to classify its corresponding cell class (1–4). 
## How to install
1. Clone the repository
    ```
    git clone git@github.com:Kirita74/NYCU-Computer-Vision-2025-Spring-HW3.git
    cd NYCU-Computer-Vision-2025-Spring-HW3
    ```
2. Create and activate conda environment
    ```
    conda env create -f environment.yml
    conda activate CV
    cd code
    ```
3. Download the dataset
    - Download the dataset form the provided [Link](https://drive.google.com/file/d/1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u/view)
    - Place it in the following structure
    ```
    .
    ├── code
    |   ├── COCOJson.py
    │   ├── utils.py
    │   ├── main.py
    │   ├── model.py
    │   └── dataset.py
    ├── data
    │   ├── train
    |   ├── test-relese
    |   └── test_image_name_to_ids.json
    ├── environment.yml
    │   .
    │   .
    │   .
    ```
## 
- Generate train json
    ```
    python3 COCOJson.py DATAPATH
    ```
- Train Model
    ```
    python3 main.py MODE DATAPATH [--num_epochs EPOCH] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--decay DECAY] [--eta_min ETA_MIN] [--pretrained_weight_path PRETRAINED_WEIGHT_PATH] [--save_path SAVE_PATH] [--log_dir LOG_DIR] [--mask_threshold MASK_THRESHOLD]
    ```
    Example:
    ```
    python3 main.py "train" ../data --num_epochs 15 --batch_size 2 --learing_rate 1e-4 --decay 1e-5 --eta_min 1e-6 --pretraind_weight_path pretrained_model.pth --save_path save_model.pth --log_dir logs
    --mask_threshold 0.5
    ```
- Test Model
    Example:
    ```
    python3 main.py "test" --pretraind_weight_path pretrained_model.pth --mask_threshold 0.7
    ```

## Performance snapshot
### Training Parameter Configuration
| Parameter                      | Value                                                                      |
|-------------------------------|----------------------------------------------------------------------------|
| **Model**                     | `ResNext 50`                                                                |
| **RPN Anchor sizes**          | (4,), (8,), (16,), (32,), (64,)                                            |
| **RPN Anchor aspect ratios**  | (0.5, 1.0, 2.0) × 5                                                         |
| **Box ROI Pooling feature map**     | `['0', '1', '2', '3']`                                                     |
| **Box ROI pooling output size**     | 7 × 7                                                                      |
| **Box ROI pooling sampling ratio**  | 2                                                                          |
| **Mask ROI Pooling feature map**     | `['0', '1', '2', '3']`                                                     |
| **Mask ROI pooling output size**     | 14 × 14                                                                      |
| **Mask ROI pooling sampling ratio**  | 4                                                                          |
| **Optimizer**                 | `AdamW`                                                                    |
| **Learning Rate**             | 1e-4                                                                       |
| **Weight Decay**              | 5e-5                                                                       |
| **Scheduler**                 | `CosineAnnealingLR`                                                        |
| **T_max**                     | 30                                                                        |
| **Epochs**                    | 30                                                                         |
| **Batch Size**                | 1                                                                          |

### Training Curve
- Epoch loss
    ![Image](image/train_loss.png)
- Mean Average Precision
    ![Image](image/mAP.png)

### Perfomance
||Accuracy(%)|
|----------|--|
|Public mAP|33|
|Public test|32|