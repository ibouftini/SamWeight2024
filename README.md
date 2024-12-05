<div align="center">
<h2>SamWeight: Optimizing Breast Cancer Detection with AUC Reshaping Techniques</h2>

[Imade Bouftini](github.com/ibouftini), [Ilyas Bounoua](), [Sacha Bouchez-Delotte](mailto:Sacha.bouchez-delotte@hera-mi.com)

[Ecole Centrale de Nantes](https://www.ec-nantes.fr/) 
</div>


<div align="center">
  <h3>Table of Contents</h3>
  <p style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin: 0;">
    <a href="#introduction">📖 Introduction</a>
    <a href="#objectives">🎯 Objectives</a>
    <a href="#methods">⚙️ Methods</a>
    <a href="#results">📊 Results</a>
    <a href="#discussion">💬 Discussion</a>
    <a href="#references">🔗 References</a>
  </p>
</div>




## **Introduction**

Breast cancer is a leading cause of mortality among women globally, emphasizing the critical role of early detection through mammography. In medical imaging, class imbalances and high misclassification costs (e.g., false negatives) pose significant challenges. Unfortunately, traditional performance metrics, such as the Area Under the Receiver Operating Characteristic (AU-ROC), fail to focus on critical regions of interest along the curve.

This project addresses these issues using a *Sample Weighting* technique to improve the sensitivity and specificity of deep learning models in detecting malignant regions in mammographic images [<a href="#ref-1">1</a>].

### **Objectives**
1. Develop and implement AUC Reshaping techniques to optimize model sensitivity at high-specificity thresholds.
2. Integrate sample weighting into a fine-tuned deep learning model to emphasize misclassified samples in critical ROC regions.
3. Evaluate the method quantitatively (e.g., sensitivity, specificity) and qualitatively (e.g., visual heatmaps of detected regions).

## **Methods**

### **Sample Weighting and AUC Reshaping**
To enhance model focus on critical regions, we employ an adaptive weighting mechanism informed by AUC Reshaping principles. This is mathematically formulated as follows:

Let $\( y_i \in \{0, 1\} \)$ denote the true label and $\( p_i \)$ the predicted probability for sample $\( i \)$. The reshaped loss function is:

$$\mathcal{L} = - \sum_{i=1}^N \left[ y_i \log(p_i - b_i) + (1 - y_i) \log(1 - p_i + b_i) \right],$$

where $\( b_i \)$ represents the boosting value, defined as:

$$b_i =
\begin{cases}
n, & \text{if } y_i = 1 \text{ and } p_i < \theta_{\text{max}}, \\
0, & \text{otherwise.}
\end{cases}
$$

Here:
- $\( \theta_{\text{max}} \)$ is the high-specificity threshold.
- $\( n \)$ is the boosting factor, modulating the emphasis on misclassified positive samples.

The AUC Reshaping function selectively modifies the ROC curve within a *Region of Interest (ROI)*, typically at high-specificity thresholds (e.g., 0.95 or 0.98). By iteratively boosting sample weights, the function reduces false negatives without significantly increasing false positives.

### **The predefined model**
The model we are going to work on is a state-of-the-art ResNet-22 [<a href="#ref-2">2</a>] with CBAM attention layers [<a href="#ref-3">3</a>]. Patches (small portions of a mammogram) containing masses and calcifications will be used to train this model [<a href="#ref-4">4</a>].

## **Results**

## **Discussion**


## **References**
<a id="ref-1"></a>[1] [Nature Research Article on Breast Cancer Detection](https://www.nature.com/articles/s41598-023-48482-x).

<a id="ref-2"></a>[2] [He, K., et al. (2015). "Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385).

<a id="ref-3"></a>[3] [Woo, S., et al. (2018). "CBAM: Convolutional Block Attention Module"](https://arxiv.org/abs/1807.06521).

<a id="ref-4"></a>[4] [William Lotter, Greg Sorensen, and David Cox. "A Multi-Scale CNN and Curriculum Learning Strategy for Mammogram Classification"](https://arxiv.org/abs/1707.06978).
