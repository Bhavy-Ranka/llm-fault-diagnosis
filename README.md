# llm-fault-diagnosis
Large Language Model framework for machine fault diagnosis using FFT and statistical features from time-series vibration data.

This repository is the  implementation of:

**FD-LLM: Large Language Model for Fault Diagnosis of Machines**

[![arXiv](https://img.shields.io/badge/arXiv-2412.01218-b31b1b.svg)](https://arxiv.org/abs/2412.01218)

Dataset Used in this :
(https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets)

## 2. Final Evaluation Performance (4 epoch)

| Metric | Value |
| :--- | :---: |
| **Accuracy** | 0.7595 |
| **Precision** | 0.7929 |
| **Recall** | 0.7595 |
| **F1-Score** | 0.7499 |

---

## 3. Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **NO** | 0.98 | 1.00 | 0.99 | 103 |
| **IRF** | 0.63 | 1.00 | 0.77 | 89 |
| **ORF** | 0.53 | 0.44 | 0.48 | 89 |
| **REF** | 1.00 | 0.56 | 0.72 | 89 |
| **Accuracy** | | | 0.76 | 370 |
| **Macro Avg** | 0.79 | 0.75 | 0.74 | 370 |
| **Weighted Avg** | 0.79 | 0.76 | 0.75 | 370 |

## 4. Confusion Matrix

| True \ Predicted | NO | IRF | ORF | REF |
| :--- | :---: | :---: | :---: | :---: |
| **NO** | 103 | 0 | 0 | 0 |
| **IRF** | 0 | 89 | 0 | 0 |
| **ORF** | 0 | 50 | 39 | 0 |
| **REF** | 2 | 3 | 34 | 50 |
