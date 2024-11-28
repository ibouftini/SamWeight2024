# SamWeight: Sample Weighting Techniques for Breast Cancer Detection in Mammography

## Introduction

### Background
Breast cancer screening relies critically on accurate mammographic image analysis. Deep learning algorithms present promising opportunities for enhancing diagnostic sensitivity while maintaining specificity. Current challenges include minimizing false-negative classifications, which could potentially compromise patient outcomes.

### Problem Statement
Medical image classification algorithms must optimize sensitivity to detect potential malignancies, necessitating sophisticated techniques to reduce misclassification of negative samples without excessive false-positive rates.

### Research Objectives
1. Explore advanced sample weighting strategies in deep learning
2. Implement an Area Under the Curve (AUC) reshaping approach
3. Develop an improved breast cancer detection model using mammographic patches

## Methods

### Experimental Design
#### Model Architecture
- Base Model: ResNet-22 
- Attention Mechanism: CBAM (Convolutional Block Attention Module)
- Input: Mammographic patches containing masses and calcifications

#### Sample Weighting Technique
Sample weighting in deep learning involves dynamically adjusting individual sample contributions during training. The proposed approach focuses on an AUC reshaping strategy to preferentially emphasize informative samples.

#### Mathematical Formulation
Let $W = \{w_1, w_2, ..., w_n\}$ represent sample weights, where:

$w_i = f(x_i, y_i)$

- $x_i$: Feature representation of sample $i$
- $y_i$: Ground truth label
- $f()$: Weight assignment function derived from AUC reshaping principles

#### Training Protocol
1. Patch extraction from mammographic images
2. Sample weight initialization
3. Iterative model training with dynamically adjusted weights
4. Evaluation of classification performance metrics

### Data Preprocessing
- Data Source: Mammographic image dataset
- Preprocessing: 
  - Patch segmentation
  - Normalization
  - Augmentation techniques

## Results

*Placeholder for experimental results*

## Discussion

*Placeholder for interpretation and implications*

## References
1. Bhat, S., et al. (2023). "AUCReshaping: improved sensitivity at high-specificity". Scientific Reports, 13(1).
2. He, K., et al. (2015). Deep Residual Learning for Image Recognition.
3. Lotter, W., Sorensen, G., & Cox, D. (2017). A Multi-Scale CNN and Curriculum Learning Strategy for Mammogram Classification.
4. Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module.
