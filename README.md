
# MaskCon with Attribute-Guided Similarity

Please consider citing our paper and starring the repository if you find this useful.  
We kindly request that before raising an issue, reflect on the value of unpaid open-source contributions. Respect and acknowledgment go a long way in sustaining community efforts.

---

##  Preparation

Ensure you have the following Python packages installed:

```bash
pip install torch tqdm wandb
```

---

##  Usage

### Example runs on ImageNet, SOP, and Cars datasets:

```bash
# ImageNet32 (MaskCon-CoIns)
python main.py --dataset imagenet32 --data_path ../datasets/imagenet32 --wandb_id imagenet32 --K 16384 --m 0.99 --t0 0.1 --t 0.05 --w 1.0 --mode coins --gpu_id 0 --epochs 100 --aug_q strong --aug_k weak --batch_size 256

# Stanford Online Products (MaskCon)
python main.py --dataset sop_split1 --data_path ../datasets/Stanford_Online_Products --wandb_id Stanford_Online_Products_Split1 --t0 0.1 --t 0.1 --w 0.8 --mode maskcon --gpu_id 0 --aug_q strong --aug_k weak

# Cars196 (MaskCon)
python main.py --dataset cars196 --data_path ../datasets/StanfordCars --wandb_id StanfordCars --t 0.1 --w 1.0 --mode maskcon --gpu_id 1 --aug_q strong --aug_k weak
```

---

##  Visualizations

### Feature Space Visualization:

- **t-SNE** on CIFARtoy test set:  
  `wandb` automatically logs visualizations.

### Top-k Retrieval:

- Example: **Top-10** retrieved images on Cars196 test set.

---

##  Reproducing CIFAR Experiments
 Cifartoy good split
```bash
python main.py \
  --dataset cifartoy \
  --data_path ./datasets \
  --wandb_id cifartoy_good \
  --mode maskcon \
  --gpu_id 0 \
  --aug_q strong --aug_k weak \
  --t0 0.1 --t 0.05 --w 1.0 \
  --epochs 50 --batch_size 256
>  Make sure you have a [Weights & Biases](https://wandb.ai/site) account configured.  
```
 CIFAR-100
 ```
 python main.py \
  --dataset cifar100_20 \
  --data_path ./datasets \
  --wandb_id cifar100_run \
  --mode maskcon \
  --gpu_id 0 \
  --aug_q strong --aug_k weak \
  --t0 0.1 --t 0.05 --w 1.0 \
  --epochs 100 --batch_size 256
```
---

##  Using Our Novelty Dataset (CUB-200-2011)

### 1. Dataset Download

Download the official dataset via either:

- [Kaggle](https://www.kaggle.com/datasets/arslankhanofficial/cub200birds)
- [Caltech Official Site](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

### 2. Directory Structure

Create a `CUB_200_2011` folder with the following internal structure:

```
CUB_200_2011/
├── attributes/
├── images/
├── parts/
├── attributes.txt
├── bounding_boxes.txt
├── classes.txt
├── image_class_labels.txt
├── images.txt
├── README
├── train_test_split.txt
```

---

### 3. Run Commands

```bash
python3 -u main.py   --dataset CUB   --data_path ./data/CUB_200_2011   --mode maskcon  --w 1.0   --t 0.5   --epochs 100   --arch resnet50   --wandb_id cs24mtech14011

```

Change `--num_classes` as follows:
| Class Split | `args.num_classes` |
|-------------|---------------------|
| 20 classes  | 4                   |
| 50 classes  | 11                  |
| 200 classes | 34                  |

>  For using different ResNet backbones, pass the `--arch` flag (`resnet18`, `resnet32`, `resnet50`).

---

### 4. Modify Class Selection

In `datasets/CUB.py`, set:

```python
self.selected_classes = [i for i in range(1, 21)]  # for 20 classes
# for 50 classes: range(1, 51)
# for 200 classes: range(1, 201)
```

---

### 5. Change Similarity Metric

By default, **Euclidean similarity** is used for attribute guidance.  
To switch to **Hamming similarity**, go to `utils/similarity.py` and:

- Comment out the Euclidean section
- Uncomment the Hamming section

---

##  Coarse Label Usage

Coarse labels for **ImageNet** and **SOP** are provided in the `coarse_labels/` directory.

- For **SOP**, copy `coarse_labels/sop_split1` and `sop_split2` into your dataset folder.
- For **ImageNet32**, copy the following into your dataset path:
  ```
  coarse_labels/imagenet32/imagenet32_to_fine.json
  coarse_labels/imagenet32/imagenet_fine_to_coarse.json
  ```



