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
python train.py --dataset data/$YOUR DATASET NAME$ --sam_pretrain pretrain/$SAM CHECKPOINT$
```

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
