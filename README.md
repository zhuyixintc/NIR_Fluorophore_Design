# NIR_Fluorophore_Design
A dual-module deep learning system to pre-screen near-infrared (NIR) fluorophores—predicting absorption, emission, and PLQY before experiments, via knowledge transfer.

<p align="left">
  <img src="https://github.com/user-attachments/assets/d3ba0c2a-d380-46b2-9463-5b2cebf5ccce" width="300" alt="overview">
</p>

## Requirements
- Pyhton=3.9 
- CUDA=11.2
- TensorFlow=2.10
- RDKit, NumPy, pandas, scikit-learn, tqdm

## Installation
- conda create -n nirfluoro python=3.9 -y
- conda activate nirfluoro
- conda install -c conda-forge rdkit=2022.09.5 -y
- pip install tensorflow==2.10.* numpy pandas scikit-learn tqdm

## Code Structure
```text
NIR_Fluorophore_Design/
├─ core/
│  ├─ data_processing.py
│  └─ model_transformer.py
├─ fine_tune.py
├─ pre_train.py
└─ README.md
```
## Workflow
- Your own data
- python pre_train.py
- python fine_tune.py

## UI download
- Link: https://pan.baidu.com/s/1wSw-tA-zLftS6x3AsOjS2Q

## Acknowledgement
This work is partially built on MG-BERT and uses the ChemFluor as initial dataset. We deeply grateful to the authors for making their code and data publicly available.

The dataset and trained models will be made available upon publication.
