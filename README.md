# llm-fault-diagnosis
Large Language Model framework for machine fault diagnosis using FFT and statistical features from time-series vibration data.

This repository is the  implementation of:

**FD-LLM: Large Language Model for Fault Diagnosis of Machines**

[![arXiv](https://img.shields.io/badge/arXiv-2412.01218-b31b1b.svg)](https://arxiv.org/abs/2412.01218)

Dataset Used in this :
(https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets)


# Evaluation Results

### 1. Overall Performance

| Metric | Value |
| :--- | :---: |
| **Accuracy** | 0.9216 |
| **Precision** | 0.9265 |
| **Recall** | 0.9216 |
| **F1-Score** | 0.9210 |

### 2. Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **NO** | 0.88 | 1.00 | 0.94 | 103 |
| **IRF** | 0.89 | 0.96 | 0.92 | 89 |
| **ORF** | 0.99 | 0.84 | 0.91 | 89 |
| **REF** | 0.95 | 0.88 | 0.91 | 89 |
| **Macro Avg** | 0.93 | 0.92 | 0.92 | 370 |
| **Weighted Avg** | 0.93 | 0.92 | 0.92 | 370 |

### 3. Confusion Matrix

| True \ Predicted | NO | IRF | ORF | REF |
| :--- | :---: | :---: | :---: | :---: |
| **NO** | 103 | 0 | 0 | 0 |
| **IRF** | 1 | 85 | 0 | 3 |
| **ORF** | 3 | 10 | 75 | 1 |
| **REF** | 10 | 0 | 1 | 78 |
