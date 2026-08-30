# Data

Raw images are **not** committed to this repo (Kaggle ToS + repo size).
Download and place them locally as follows.

## 1. APTOS 2019 (primary training set)
- Source: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
- Download `train.csv` and `train_images/` folder.
- Place at:
  ```
  data/aptos2019/train.csv
  data/aptos2019/train_images/
  ```

## 2. IDRiD (Grad-CAM validation only)
- Source: https://idrid.grand-challenge.org/Data/
- Use only the lesion-annotated subset ("A. Segmentation").
- Place at:
  ```
  data/idrid/images/
  data/idrid/lesion_masks/
  ```

## 3. Messidor-2 (stretch goal — external validation)
- Source: https://www.adcis.net/en/third-party/messidor2/
- Place at:
  ```
  data/messidor2/images/
  data/messidor2/grades.csv
  ```

## Setup
```bash
mkdir -p data/aptos2019 data/idrid data/messidor2
# then download and unzip into the folders above
```

See `docs/decisions/0002-dataset-choice.md` for why these were chosen.
