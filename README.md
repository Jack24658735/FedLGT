[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/cs.CV-%09arXiv%3A2312.07165-red)](https://arxiv.org/abs/2312.07165)

<div align="center">
<h1>
<b>
Language-Guided Transformer for <br> Federated Multi-Label Classification
</b>
</h1>
</div>

### [Project Page](https://jack24658735.github.io/fedlgt/) | [Paper](https://arxiv.org/abs/2312.07165)




> **Language-Guided Transformer for Federated Multi-Label Classification**
> 
> **I-Jieh Liu**, Ci-Siang Lin, Fu-En Yang, Yu-Chiang Frank Wang

Official implementation of our work **Language-Guided Transformer for Federated Multi-Label Classification**.
<p align="center"><img src="assets/fig1_for_project_page.png" width="450"/></p>


## Abstract
Federated Learning (FL) is an emerging paradigm that enables multiple users to collaboratively train a robust model in a privacy-preserving manner without sharing their private data. Most existing approaches of FL only consider traditional single-label image classification, ignoring the impact when transferring the task to multi-label image classification. Nevertheless, it is still challenging for FL to deal with user heterogeneity in their local data distribution in the real-world FL scenario, and this issue becomes even more severe in multi-label image classification. Inspired by the recent success of Transformers in centralized settings, we propose a novel FL framework for multi-label classification. Since partial label correlation may be observed by local clients during training, direct aggregation of locally updated models would not produce satisfactory performances. Thus, we propose a novel FL framework of **L**anguage-**G**uided **T**ransformer (**FedLGT**) to tackle this challenging task, which aims to exploit and transfer knowledge across different clients for learning a robust global model. Through extensive experiments on various multi-label datasets (e.g., FLAIR, MS-COCO, etc.), we show that our FedLGT is able to achieve satisfactory performance and outperforms standard FL techniques under multi-label FL scenarios.

## Update
- **(2023/12/10)** Code for FedLGT is coming soon. Stay tuned!

## Framework Overview
<p align="center"><img src="assets/fig2_for_project_page.png" width="800"/></p>

## Setup
1. Please install your PyTorch version according to your CUDA version. For more details, please refer to [PyTorch](https://pytorch.org/). 
    * Sample command:
        ```
        pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
        ```

2. Run the following command to install the required packages.
    ```
    pip install -r requirements.txt
    ```
## Data Preparation
### FLAIR
1. Please refer to [FLAIR](https://github.com/apple/ml-flair) official repository for data preparation. Note that please run the script to generate the hdf5 file for FLAIR dataset. For more details, please refer to [prepare-hdf5](https://github.com/apple/ml-flair?tab=readme-ov-file#optional-prepare-the-dataset-in-hdf5) part in FLAIR.
2. Run our pre-processing commands:
    ```
    python3 ./data/build_text_feat.py
    python3 ./data/build_label_mapping.py
    ```
3. Now you should have all the data files under the folder ``data``.
### COCO/PASCAL VOC
Please refer to [C-Tran](https://github.com/QData/C-Tran) to obtain these datasets.


## Training and Evaluation
* Run the following command to perform FL training on FLAIR.
    * **Coarse-grained**
    ``` python
    # Coarse-grained 
    python fed_main.py --batch_size 16  --lr 0.0001 --optim 'adam' --layers 3  --dataset 'flair_fed' \ 
                        --use_lmt --grad_ac_step 1 --dataroot data/ --epochs 5 --n_parties 50 --comm_round 50 \ 
                        --learn_emb_type clip --agg_type fedavg --coarse_prompt_type concat --use_global_guide
    
    ```
    * **Fine-grained**
    ``` python
    # Fine-grained
    python fed_main.py --batch_size 16  --lr 0.0001 --optim 'adam' --layers 3  --dataset 'flair_fed' \ 
                    --use_lmt --grad_ac_step 1 --dataroot data/ --epochs 5 --n_parties 50 --comm_round 50 \
                    --learn_emb_type clip --agg_type fedavg --coarse_prompt_type concat --flair_fine --use_global_guide

    ```
* Evaluation
    * Follow the same comamnd as training, but add `--inference` flag. For example:
    ``` python
    # Coarse-grained
    python fed_main.py  --layers 3 --dataset 'flair_fed' \ 
                        --use_lmt --dataroot data/ --n_parties 1 \ 
                        --learn_emb_type clip --coarse_prompt_type concat --use_global_guide --inference
    
    ```


## Acknowledgement
We build our FedLGT codebase on the codebases of [C-Tran](https://github.com/QData/C-Tran) and [NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench). We sincerely thank for their wonderful works.

## Citation
If you find this useful for your research, please consider citing:
``` bibtex
@inproceedings{liu2024fedlgt,
  author    = {I-Jieh Liu and Ci-Siang Lin and Fu-En Yang and Yu-Chiang Frank Wang},
  title     = {Language-Guided Transformer for Federated Multi-Label Classification},
  booktitle = {AAAI},
  year      = {2024},
}
```

## Contact
If you have any questions about this project, please feel free to contact liujack0914@gmail.com.

