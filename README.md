# Universal Prompt-Free Segmentation for Generalized Nucleus Images (UN-SAM)
This repository is an official PyTorch implementation of the paper **"UN-SAM: Universal Prompt-Free Segmentation for Generalized Nucleus Images"** [[paper]()] submitted to IEEE Transactions on Medical Imaging.

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
## Checkpoints
* UN

## Cite
If you find our work useful in your research or publication, please cite our work:
```
@article{chen2024sam,
  title={UN-SAM: Universal Prompt-Free Segmentation for Generalized Nuclei Images},
  author={Chen, Zhen and Xu, Qing and Liu, Xinyu and Yuan, Yixuan},
  journal={arXiv preprint arXiv:2402.16663},
  year={2024}
}
```


## Acknowledgements
* [SAM](https://github.com/facebookresearch/segment-anything)
