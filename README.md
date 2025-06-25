# UN-SAM: Domain-Adaptive Self-Prompt Segmentation for Universal Nuclei Images
This repository is an official PyTorch implementation of the paper **"UN-SAM: Domain-Adaptive Self-Prompt Segmentation for Universal Nuclei Images"** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841525001549)] accepted by Medical Image Analysis.

## Dependencies
* Python 3.10
* PyTorch >= 1.10.0
* albumentations 1.5.2
* monai 1.3.0
* pytorch_lightning 1.1.0


## Code
Clone this repository into any place you want.
```bash
git clone https://github.com/CUHK-AIM-Group/UNSAM.git
cd UNSAM
mkdir data; mkdir pretrain;
```
## Quickstart 
* Train the UN-SAM with the default settings:
```python
python train.py --domain_num $NUMBER OF DOMAINS$ --size $B$ --sam_pretrain pretrain/$SAM CHECKPOINT$
```

## Data Preparation
The structure is as follows.
```
UN-SAM
├── data
│   ├── DSB-2018
|     ├── data_split.json
│   ├── MoNuSeg
|     ├── data_split.json
│   ├── TNBC
|     ├── data_split.json
│   ├── image_1024
│     ├── DSB_0000000.png
│     ├── MoNuSeg_0000000.png
│     ├── TNBC_0000000.png
|     ├── ...
|   ├── mask_1024
│     ├── DSB_0000000.png
│     ├── MoNuSeg_0000000.png
│     ├── TNBC_0000000.png
|     ├── ...   
```

## Pre-trained Model Zoo 
We provide all pre-trained models here.
| Size | Domains | Checkpoints |
|-----|-----|-----|
|UN-SAM-B| DSB-2018 + MoNuSeg + TNBC |[Google Drive](https://drive.google.com/drive/folders/1wx5LYQ5hIR53NonsDSmTm0krtnR6Iq09?usp=sharing)|
|UN-SAM-L| DSB-2018 + MoNuSeg + TNBC |TBA|
|UN-SAM-H| DSB-2018 + MoNuSeg + TNBC |TBA|


## Cite
If you find our work useful in your research or publication, please cite our work:
```
@article{chen2025sam,
  title={UN-SAM: Domain-adaptive self-prompt segmentation for universal nuclei images},
  author={Chen, Zhen and Xu, Qing and Liu, Xinyu and Yuan, Yixuan},
  journal={Medical Image Analysis},
  pages={103607},
  year={2025},
  publisher={Elsevier}
}
```


## Acknowledgements
* [SAM](https://github.com/facebookresearch/segment-anything)
