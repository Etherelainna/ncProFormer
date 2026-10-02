# ncProFormer: A CNN-enhanced Transformer framework for ncRNA coding potential prediction

ncProFormer is a deep learning–based framework for ncRNA coding potential prediction, which combines pretrained sequence embeddings with a CNN-enhanced Transformer to model both local and long-range features of RNA sequences.

---

## Environment Setup

**Python >= 3.8, < 3.11**

```text
biopython==1.79
numpy==1.24.4
pandas==1.5.3
scikit-learn==1.3.2
scipy==1.10.1
matplotlib==3.7.5
seaborn==0.13.2
transformers==4.46.3
tokenizers==0.20.3
accelerate==1.0.1
tqdm
pyyaml
requests
safetensors
huggingface-hub
```

## Pretrained Model
By default, ncProFormer uses a pretrained BigBird-based GENA-LM model for sequence embedding.

Note:
Different pretrained GENA-LM BigBird variants may be used depending on availability and computational settings. This does not affect the overall architecture of ncProFormer. Users may optionally download pretrained GENA-LM weights from the official repository and place them in a local directory. The path can be specified via configuration.

GENA-LM repository:
https://github.com/AIRI-Institute/GENA_LM

## Feature Construction
Feature construction is performed in the following order.

Step 1: Hexamer Feature Preparation
```
python Hexamer_build.py
```

Step 2: Feature Extraction
```
# training set
python feature_build.py --lnc /path/to/features_train.csv --fasta /path/to/train.fasta --hexamer /path/to/RNA_Hexamer.tsv
# validation set
python feature_build.py --lnc /path/to/features_val.csv --fasta /path/to/val.fasta --hexamer /path/to/RNA_Hexamer.tsv
# test set
python feature_build.py --lnc /path/to/features_test.csv --fasta /path/to/test.fasta --hexamer /path/to/RNA_Hexamer.tsv
```

Note:
For training on user-provided datasets, part of the handcrafted features are extracted using LncFinder (v1.1.6). These features should be computed separately and provided as input to feature_build.py.

## Model Training and Evaluation
After feature construction is completed, run the following command to train the model and evaluate its performance
```
python mean.py
```

## Pretrained Model Weights
We provide the pretrained weights of the main task model: best_main_model.pt

## Public Benchmark Dataset
The public benchmark dataset used for comparison is adopted from **CPPred**.  
Please download the dataset directly from the official CPPred website:
http://www.rnabinding.com/CPPred/








