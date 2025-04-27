<div id="top" align="center">

# Towards One-Stage End-to-End Table Structure Recognition with Parallel Regression for Diverse Scenarios

  Anyi Xiao and Cihui Yang* </br>

  [![arXiv](https://img.shields.io/badge/arXiv-2504.17522-b31b1b.svg)](http://arxiv.org/abs/2504.17522)

</div> 

## Abstract
Table structure recognition aims to parse tables in unstructured data into machine-understandable formats. Recent methods address this problem through a two-stage process or optimized one-stage approaches. However, these methods either require multiple networks to be serially trained and perform more time-consuming sequential decoding, or rely on complex post-processing algorithms to parse the logical structure of tables. They struggle to balance cross-scenario adaptability, robustness, and computational efficiency. In this paper, we propose a one-stage end-to-end table structure parsing network called TableCenterNet. This network unifies the prediction of table spatial and logical structure into a parallel regression task for the first time, and implicitly learns the spatial-logical location mapping laws of cells through a synergistic architecture of shared feature extraction layers and task-specific decoding. Compared with two-stage methods, our method is easier to train and faster to infer. Experiments on benchmark datasets show that TableCenterNet can effectively parse table structures in diverse scenarios and achieve state-of-the-art performance on the TableGraph-24k dataset. 

# Getting Started
## Installation
### Requirements
Create the environment from the environment.yml file `conda env create --file environment.yml` or install the software needed in your environment independently.
```
name: TableCenterNet
channels:
  - defaults
dependencies:
  - pip==24.2
  - python==3.8.20
  - setuptools==75.1.0
  - wheel==0.44.0
  - pip:
      - numpy==1.24.4
      - opencv-contrib-python==4.11.0.86
      - opencv-python==4.10.0.84
      - openpyxl==3.1.5
      - pandas==2.0.3
      - pillow==10.4.0
      - pycocotools==2.0.7
      - pyyaml==6.0.2
      - scipy==1.10.1
      - shapely==2.0.6
      - table-recognition-metric==0.0.4
      - tabulate==0.9.0
      - thop==0.1.1-2209072238
      - timm==0.4.12
      - torch==2.4.0
      - torchvision==0.19.0
      - tqdm==4.66.5
```

### Installing SciTSR
This package is mainly used to evaluate adjacency relationship.

```
git clone https://github.com/Academic-Hammer/SciTSR.git
cd SciTSR
python setup.py bdist_wheel
pip install dist/scitsr-0.0.1-py3-none-any.whl
```

## Preparation
### Datasets
- Doownload datasets from [Releases](https://github.com/dreamy-xay/releases/tag/v1.0).
- Put `datasets.tar.gz` in **"./datasets/"** and unzip it.
```
cd TableCenterNet/datasets
tar -zxvf datasets.tar.gz
## The './datasets/' folder should look like:
checkpoints
├─── ICDAR2013
├─── WTW
└─── TG24K
```

### Pretrained Models
- Download pretrained models from [Releases](https://github.com/dreamy-xay/TableCenterNet/releases/tag/v1.0).
- Put `checkpoints.tar.gz` in **"./checkpoints/"** and unzip it.
```
cd TableCenterNet/checkpoints
tar -zxvf checkpoints.tar.gz
## The './checkpoints/' folder should look like:
checkpoints
├─── ICDAR2013
├─── WTW
└─── TG24K
```

## Testing

We have prepared scripts for test and you can just run them, command line as following:
```
cd TableCenterNet

# Test ICDAR2013
sh scripts/test/${BACKBONE}/test_icdar2013.sh

# Test the wired tables in ICDAR2013 only
sh scripts/test/${BACKBONE}/test_icdar2013_wired.sh

# Test WTW
sh scripts/test/${BACKBONE}/test_wtw.sh

# Test TableGraph-24k
sh scripts/test/${BACKBONE}/test_tg24k.sh
```

where `${BACKBONE}` should be replaced with “dla” or “star” to indicate the use of DLA-34 or StarNet-s3 as the backbone, respectively.

## Training

We have prepared the training scripts. Note that ${BACKBONE} needs to be replaced with “dla” and “star” before running the script. The command line is as follows:
```
cd TableCenterNet

# Fine-tuning ICDAR2013
sh scripts/train/${BACKBONE}/train_icdar2013.sh

# Train WTW
sh scripts/train/${BACKBONE}/train_wtw.sh

# Train TableGraph-24k
sh scripts/train/${BACKBONE}/train_tg24k.sh
```

If you need to train a customized dataset, please convert the labels to the COCO format. Here, the WTW dataset is taken as an example. The directory of dataset are organized as following:
```
data
└── WTW
    ├── images
    └── labels
        ├──train.json
        └──test.json
```
Then configure the loading and preprocessing strategies for the dataset. Specifically, you can refer to all the configuration files under "./src/cfg/datasets/". Finally, modify the `--data` item in the training script to your customized dataset configuration.

## Evaluating

Execute the following scripts to evaluate TableCenterNet:
```
cd TableCenterNet

# Evaluate ICDAR2013
sh scripts/val/${BACKBONE}/val_icdar2013.sh

# Evaluate the wired tables in ICDAR2013 only
sh scripts/val/${BACKBONE}/val_icdar2013_wired.sh

# Evaluate WTW
sh scripts/val/${BACKBONE}/val_wtw.sh

# Evaluate TableGraph-24k
sh scripts/val/${BACKBONE}/val_tg24k.sh
```

The above scripts will infer before evaluating, or if you only need to evaluate, you can execute the following command:
```
cd TableCenterNet

python src/main.py mtable val --only_eval --label ${LABEL_PATH} --project ${PROJECT_FOLDER} --name ${EXP_NAME}

### Example: We validate the test results on the ICDAR2013 dataset
# Test first
sh scripts/test/dla/test_icdar2013_wired.sh 
# Evaluate
python src/main.py mtable val \
  --only_eval \
  --label datasets/ICDAR2013/labels/wired_test.json \
  --project Test/ICDAR2013 \
  --name dla_wired
```

After the evaluation, `evaluate_results.md` and `evaluate_results.xlsx` will be generated in the `./${PROJECT_FOLDER}/${EXP_NAME}/` folder to save the results.

## Acknowledgements
This implementation refers to [CenterNet](https://github.com/xingyizhou/CenterNet) and [LORE](http://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/LORE-TSR).