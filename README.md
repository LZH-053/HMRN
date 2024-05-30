# Hierarchical Matching and Reasoning for Multi-Query Image Retrieval
[Hierarchical Matching and Reasoning for Multi-Query Image Retrieval](https://doi.org/10.1016/j.neunet.2024.106200)
Zhong Ji, Zhihao Li, Yan Zhang, Haoran Wang, Yanwei Pang, Xuelong Li. Neural Networks 2024

PyTorch implementation of our method for multi-query image retrieval.
## Requirements
- Setup a conda environment and install some prerequisite packages including
```
conda create -n HMRN python=3.8    # Create a virtual environment
conda activate HMRN         	   # Activate virtual environment
conda install jupyter scikit-image cython opencv seaborn nltk pycairo h5py  # Install dependencies
python -m nltk.downloader all	   # Install NLTK data
```
- Please also install PyTorch 2.0.1 (or higher), torchvision and torchtext.

## Data
Refer [DrillDown](https://github.com/uvavision/DrillDown) to download images, features and annotations of Visual Genome.
Download `DrillDown/data/caches` from [DrillDown](https://github.com/uvavision/DrillDown) and put the directory under `HMRN/data`
```
data
├── caches
│   ├── raw_test.txt 
│   ├── vg_attributes_vocab_1000.txt
│   ├── vg_objects_vocab_1600.txt 
│   ├── vg_objects_vocab_2500.txt 
│   ├── vg_relations_vocab_500.txt 
│   ├── vg_scenedb.pkl                 # auto-generated upon initial execution
│   ├── vg_test.txt 
│   ├── vg_train.txt 
│   ├── vg_val.txt 
│   ├── vg_vocab_14284.pkl  
│   
├── vg
│   ├── global_features 
│   │      ├── xxx.npz
│   │      └── ...
│   │ 
│   ├── region_36_final   
│   │      ├── xxx.npz
│   │      └── ...
│   │ 
│   └── rg_jsons 
│   │      ├── xxx.json
│   │      └── ...
│   │ 
│   └── sg_xmls
│   │      ├── xxx.xml
│   │      └── ...
│   │ 
│   └── VG_100K
│   │      ├── xxx.jpg
│   │      └── ...
│   │ 
│   └── VG_100K_2
│   │      ├── xxx.jpg
│   │      └── ...
│
```

## Training
- Train HMRN I-T
```
python train.py --cross_attention_direction I-T
```
- Train HMRN T-I
```
python train.py --cross_attention_direction T-I
```

## Evaluation
Please rename the saved HMRN I-T and HMRN T-I model as `model_best.pth.tar`, then refer to the following order to run the evaluation script.
- Evaluate HMRN I-T or HMRN T-I 
```
python evaluation_individual.py
```
- Evaluate HMRN ensemble
```
python evaluation_ensemble.py
```

## Acknowledgment
This codebase is partially based on [DrillDown](https://github.com/uvavision/DrillDown) and [SGRAF](https://github.com/Paranioar/SGRAF).

## Citation
If you find our paper/code useful, please cite the following paper:
```
@article{ji2024hierarchical,
  title={Hierarchical matching and reasoning for multi-query image retrieval},
  author={Ji, Zhong and Li, Zhihao and Zhang, Yan and Wang, Haoran and Pang, Yanwei and Li, Xuelong},
  journal={Neural Networks},
  pages={106200},
  year={2024},
  publisher={Elsevier}
}
```
