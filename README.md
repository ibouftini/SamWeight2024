
# **SamWeight: Optimizing Breast Cancer Detection with AUC Reshaping Techniques**

## **Introduction**

### **Overview**
Breast cancer is a leading cause of mortality among women globally, emphasizing the critical role of early detection through mammography. This project leverages *Sample Weighting* and *AUC Reshaping* techniques to improve the sensitivity and specificity of deep learning models in detecting malignant regions in mammographic images.

### **Challenges in Detection**
In medical imaging, class imbalances and high misclassification costs (e.g., false negatives) pose significant challenges. Traditional performance metrics, such as the Area Under the Receiver Operating Characteristic (AU-ROC), fail to focus on critical regions of interest along the curve. This project addresses these issues using a novel AUC Reshaping strategy.

### **Objectives**
1. Develop and implement AUC Reshaping to optimize high-specificity sensitivity.
2. Incorporate sample weighting during training to emphasize misclassified samples in critical regions of the ROC curve.
3. Improve breast cancer detection performance using deep learning architectures tailored for mammographic patches.

---

## **Methods**

### **Experimental Design**

#### **Model Architecture**
- **Base Model**: ResNet-22
- **Attention Mechanism**: Convolutional Block Attention Module (CBAM)
- **Input**: Segmented patches from mammographic images, targeting masses and calcifications.

#### **Sample Weighting and AUC Reshaping**
To enhance model focus on critical regions, we employ an adaptive weighting mechanism informed by AUC Reshaping principles. This is mathematically formulated as follows:

Let \( y_i \in \{0, 1\} \) denote the true label and \( p_i \) the predicted probability for sample \( i \). The reshaped loss function is:

\[\mathcal{L} = - \sum_{i=1}^N \left[ y_i \log(p_i - b_i) + (1 - y_i) \log(1 - p_i + b_i) \right],
\]

where \( b_i \) represents the boosting value, defined as:

\[b_i =
\begin{cases}
n, & \text{if } y_i = 1 \text{ and } p_i < \theta_{\text{max}}, \\
0, & \text{otherwise.}
\end{cases}
\]

Here:
- \( \theta_{\text{max}} \) is the high-specificity threshold.
- \( n \) is the boosting factor, modulating the emphasis on misclassified positive samples.

#### **Mathematical Justification**
The AUC Reshaping function selectively modifies the ROC curve within a *Region of Interest (ROI)*, typically at high-specificity thresholds (e.g., 0.95 or 0.98). By iteratively boosting sample weights, the function reduces false negatives without significantly increasing false positives.

### **Training Protocol**
1. **Patch Extraction**:
   - Extract ROI patches from mammograms using segmentation.
2. **Preprocessing**:
   - Normalize intensity values.
   - Apply augmentations (e.g., rotation, flipping).
3. **Initialization**:
   - Assign initial weights to all samples.
4. **Iterative Weight Adjustment**:
   - Boost weights for misclassified samples at \( \theta_{\text{max}} \) during training.
   - Optimize using the weighted cross-entropy loss function.
5. **Evaluation**:
   - Measure sensitivity at fixed specificity thresholds (e.g., 95%, 98%).

---

## **Results**
*Pending implementation*. Preliminary expectations include:
- Improved sensitivity in high-specificity regions.
- Reduced false-negative rates without compromising specificity.

---

## **Discussion**
The reshaping and weighting techniques align the model's focus on diagnostically critical regions, addressing imbalances inherent in medical datasets. This approach has the potential to significantly enhance clinical decision-making processes in mammography.

---

## **References**
1. Bhat, S., et al. (2023). "AUCReshaping: improved sensitivity at high-specificity". *Scientific Reports*.
2. He, K., et al. (2015). "Deep Residual Learning for Image Recognition".
3. Woo, S., et al. (2018). "CBAM: Convolutional Block Attention Module".
