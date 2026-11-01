# MongoDB NoSQL Injection Detection: Dataset Construction, Model Evaluation, and Explainable AI Analysis

This repository contains the code, dataset, and experimental results from a research study investigating machine learning and deep learning approaches for MongoDB NoSQL injection detection.

The study introduces a balanced and structurally consistent MongoDB injection dataset, evaluates multiple machine learning and deep learning architectures, performs cross-dataset generalization experiments, and applies explainable AI techniques to analyze model behavior.

---

## Overview

NoSQL databases have become a core component of modern applications due to their scalability and flexible data models. However, their flexible query structures also introduce security risks, including NoSQL injection attacks that can bypass authentication, expose sensitive information, alter database behavior, or disrupt application functionality.

Research on MongoDB injection detection remains limited by the scarcity of publicly available datasets, class imbalance in existing resources, and a limited exploration of modern deep learning techniques.

To address these limitations, this study:

- Develops a balanced dataset containing benign and malicious MongoDB queries.
- Evaluates multiple machine learning models to establish detection baselines.
- Investigates deep learning architectures capable of learning directly from raw queries.
- Performs cross-dataset experiments to assess model generalization.
- Applies explainable AI techniques to improve transparency and interpretability.

---

## Dataset

The dataset contains a total of **1,600 MongoDB queries** evenly distributed across benign and malicious classes.

| Category | Count |
|-----------|--------|
| Benign Queries | 800 |
| Malicious Queries | 800 |
| Total Queries | 1600 |

### Collections Covered

- login
- users
- products
- orders

### Injection Types

The malicious dataset includes six categories of MongoDB injection attacks:

- Tautologies
- Piggybacked Queries
- JavaScript Injection
- Union Queries
- Blind Injection
- Time-Based Injection

### MongoDB Operations

The dataset covers 19 MongoDB operations including:

- find
- findOne
- insertOne
- insertMany
- updateOne
- updateMany
- deleteOne
- deleteMany
- aggregate
- bulkWrite
- replaceOne
- findAndModify
- findOneAndUpdate

and several additional MongoDB operations to provide broad operational coverage.

---

## Repository Structure

```text
.
├── Code/
│
├── Dataset/
│
└── README.md
```

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-Learn
- XGBoost
- SHAP
- LIME
- Pandas
- MongoDB

---

## Experimental Configuration

### Deep Learning Training Configuration

| Parameter | Transformer Models | BiLSTM-Based Models |
|------------|------------|------------|
| Learning Rate | 1e-5 | 3e-4 |
| Batch Size | 16 | 32 |
| Maximum Epochs | 10 | 15 |
| Early Stopping | Patience = 3 | Patience = 3 |
| Dropout | 0.2 | 0.3 (BiLSTM), 0.5 (Classifier) |
| Optimizer | AdamW | AdamW |
| Loss Function | Cross-Entropy Loss | Cross-Entropy Loss |

### Hardware Environment

| Component | Specification |
|------------|------------|
| GPU | NVIDIA RTX A4000 |
| CUDA Version | 12.8 |
| RAM | 32 GB |
| Storage | 1 TB SSD |
| Operating System | Ubuntu 20.04.6 LTS |

---

## Machine Learning Models

The following classical machine learning models were evaluated:

- Decision Tree
- Random Forest
- AdaBoost
- Support Vector Machine (SVM)
- k-Nearest Neighbors (kNN)
- XGBoost
- Multi-Layer Perceptron (MLP)

A binary feature extraction pipeline consisting of 19 MongoDB injection indicators was used to generate model inputs.

---

## Deep Learning Models

The following deep learning architectures were evaluated:

- BiLSTM with Attention
- BERT
- TextCNN + BiLSTM
- DistilBERT + TF-IDF AutoEncoder

These architectures represent sequential, contextual, and hybrid approaches for NoSQL injection detection.

---

## Results

### Machine Learning Performance

| Model | Accuracy (%) |
|---------|---------|
| AdaBoost | 95.00 |
| MLP | 95.00 |
| Random Forest | 94.69 |
| SVM | 94.69 |
| Decision Tree | 94.38 |
| XGBoost | 94.06 |
| kNN | 93.75 |

### Deep Learning Performance

| Model | Accuracy (%) |
|---------|---------|
| DistilBERT + TF-IDF AutoEncoder | 98.44 |
| BERT | 97.81 |
| TextCNN + BiLSTM | 96.56 |
| BiLSTM | 96.25 |

The DistilBERT + TF-IDF AutoEncoder hybrid achieved the highest overall performance, while transformer-based models consistently demonstrated strong detection capability and interpretability.

---

## Cross-Dataset Evaluation

Cross-dataset experiments were conducted using a previously published MongoDB injection dataset and the dataset developed in this study.

Results showed that models trained on the balanced dataset introduced in this work generalized substantially better across datasets than models trained on previous imbalanced resources, highlighting the importance of dataset diversity, structural consistency, and balanced class representation.

---

## Explainability Analysis

To improve transparency and understand model decision-making, explainability techniques were applied to both machine learning and deep learning models.

### Classical Machine Learning

- Tree Feature Importance
- Permutation Importance

<p align="center">
  <img src="images/permutation_importance.png" width="800">
</p>

<p align="center">
  <img src="images/tree_importance.png" width="800">
</p>

### Deep Learning

- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-Agnostic Explanations)

The analysis showed that transformer-based models not only achieved the highest predictive performance but also produced the most coherent and interpretable explanations.

---

## Key Contributions

- Developed a balanced MongoDB injection dataset containing 1,600 queries.
- Included six categories of MongoDB injection attacks.
- Evaluated both machine learning and deep learning detection approaches.
- Performed cross-dataset generalization experiments.
- Applied explainable AI techniques to analyze model behavior.
- Demonstrated the effectiveness of transformer-based architectures for MongoDB injection detection.

---
## How to Run

### 1. Clone Repository

```bash
git clone https://github.com/HardiMatholia/Linguistic_Steganalysis.git
```

### 2. Change directory
```bash
cd MongoDB_Injection_Detection
```

### 3. Setup Instructions

1. **Update Dataset Paths**  
   - Modify the path to the dataset in the script to match the dataset location on your machine.

2. **Install Dependencies**  
   - Make sure all required Python packages are installed 

### 4. Train and Evaluate Models

```bash
python Deep Learning Models/bert_injection_detection.py
```
---
