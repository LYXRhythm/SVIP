# SVIP
# Learning Source-free Domain Adaptation for Visible-Infrared Person Re-Identification

Official implementation of the paper **"Learning Source-free Domain Adaptation for Visible-Infrared Person Re-Identification"**.

## Abstract
In this paper, we investigate source-free domain adaptation (SFDA) for visible-infrared person re-identification (VI-ReID), aiming to adapt a pre-trained source model to an unlabeled target domain without access to source data. To address this challenging setting, we propose a novel learning paradigm, termed Source-Free Visible-Infrared Person Re-Identification (SVIP), which fully exploits the prior knowledge embedded in the source model to guide target domain adaptation. The proposed framework comprises three key components specifically designed for the source-free scenario: 1) a Source-Guided Contrastive Learning (SGCL) module, which leverages the discriminative feature space of the frozen source model as a reference to perform contrastive learning on the unlabeled target data, thereby preserving discrimination without requiring source samples; 2) a Residual Transfer Learning (RTL) module, which learns residual mappings to adapt the target model’s representations while maintaining the knowledge from the source model; and 3) a Structural Consistency-Guided Cross-modal Alignment (SCCA) module, which enforces reciprocal structural constraints between visible and infrared modalities to identify reliable cross-modal pairs and achieve robust modality alignment without source supervision. Extensive experiments on benchmark datasets demonstrate that SVIP substantially enhances target domain performance and outperforms existing unsupervised VI-ReID methods under source-free settings.
![SVIP](figures/framework.png)

## Requirements
- python 3.8.13
- torch 1.8.0
- torchvision 0.9.0
- scikit-learn 1.2.2

## Quick Start 
In the following instructions, we take the transfer setting SYSU-MM01 → RegDB as a typical example.
### Dataset Preprocessing
Please change the dataset path to your own path in the `prepare_regdb.py`.
```shell
python prepare_regdb.py
```

### Source Model
Put your source model trained on SYSU-MM01 into `./pretrained` and name it `resnet50-sysumm01.pth`.

### Training
```shell
bash run_train_regdb.sh
```

### Testing
```shell
bash test_regdb.sh
```

## Generate Data for Weather Setting
To evaluate under the weather degradation setting, generate the required traget domain data with the following command. The example images in `./data_corruptions/examples` will be transformed and saved to `./data_corruptions/examples_results`.
```shell
python make_corrupt.py
```

## Citation
If our work is helpful for your research, please consider citing:
```
@inproceedings{li2025learning,
  title={Learning Source-Free Domain Adaptation for Visible-Infrared Person Re-Identification},
  author={Li, Yongxiang and Feng, Yanglin and Sun, Yuan and Peng, Dezhong and Peng, Xi and Hu, Peng},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025}
}