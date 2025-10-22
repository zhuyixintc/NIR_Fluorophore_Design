# Near_Infrared_Fluorophore_Design
A dual-module deep learning system to pre-screen near-infrared (NIR) fluorophores—predicting absorption, emission, and PLQY before experiments, via knowledge transfer.

<p align="left">
  <img src="https://github.com/user-attachments/assets/c4ec3225-e513-4a7f-955c-50db08197c34" width="300" alt="overview">
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
Near_Infrared_Fluorophore_Design/
├─ data/
│  └─ additional_data.csv
├─ molecular_generation/
│  └─ generate_molecules.py
├─ property_prediction/
│  ├─ models/
│  │  ├─ DataProcessing.py
│  │  ├─ Encoder.py
│  │  ├─ EncoderLayer.py
│  │  ├─ ModelFinetune.py
│  │  ├─ ModelPretrain.py
│  │  └─ MultiHeadAttention.py
│  ├─ pre_train.py
│  └─ fine_tune.py
└─ README.md
```
## Workflow
- Your own data
- python property_prediction/pre_train.py
- python property_prediction/fine_tune.py
- python molecular_generation/generate_molecules.py
- Screening

## UI download
- Link: https://pan.baidu.com/s/1wSw-tA-zLftS6x3AsOjS2Q
- Password: 7777 

## Acknowledgement
This program is partially built on MG-BERT. We are grateful for their open-source codes.

