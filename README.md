# MLP vs CNN for Image Classification on CIFAR-10

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)

**Author:** Dasari Lohith  
**Roll Number:** 24110100  
**Assignment:** Assignment 4 (Neural Networks & Computer Vision)

---

## 📌 Project Overview

This repository contains an in-depth empirical comparison between **Multi-Layer Perceptrons (MLPs)** and **Convolutional Neural Networks (CNNs)** for image classification on the **CIFAR-10** benchmark dataset. 

The project investigates the core architectural principles that govern neural networks for visual data, demonstrating through controlled experiments:
1. Why MLPs struggle with computer vision tasks (destruction of 2D spatial locality upon flattening, absence of translation equivariance, and susceptibility to overfitting).
2. Why CNNs naturally excel at visual tasks (local receptive fields, parameter sharing, hierarchical feature extraction from edges to shapes, and translation invariance via pooling).
3. The specific roles of **model capacity**, **pixel permutation**, **spatial translation**, and **spatial pooling**.

---

## 📊 Dataset: CIFAR-10

- **Classes (10):** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
- **Images:** 60,000 $32 \times 32$ RGB images (50,000 training, 10,000 test).
- **Preprocessing:** Normalized across RGB channels with mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)` to scale inputs into the range `[-1, 1]`.

---

## 🔬 Experimental Breakdown & Key Findings

### Part 1: Multi-Layer Perceptrons (MLP)

#### 1. Baseline MLP Architecture
- **Input:** $3 \times 32 \times 32$ flattened into a 1D vector of dimension $3072$.
- **Layers:** `Linear(3072 -> 512)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(512 -> 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(256 -> 10)`.
- **Observations:** Training loss steadily decreases while validation accuracy quickly plateaus (~45–55%), indicating rapid overfitting due to the dense parameterization.

#### 2. Model Capacity Experiment (Bias-Variance Tradeoff)
Three capacity variants were trained and benchmarked:
- **Small Model (`[128]`):** Suffers from underfitting. Lacks capacity to learn complex visual boundaries, resulting in low training and test accuracy.
- **Medium Model (`[256, 256]`):** Optimal balance between model complexity and generalization.
- **Large Model (`[512, 256, 128]`):** Memorizes training examples; training accuracy increases significantly while test accuracy degrades (severe overfitting).

#### 3. Loss of Spatial Geometry via Flattening
- Flattening converts structured 2D images into 1D arrays of size 3072.
- Pixels that were vertical or horizontal neighbors lose their topological adjacency.
- Visualization confirms that the flattened 1D signal appears as random noise without spatial context.

#### 4. The Pixel Shuffling / Permutation Experiment
To prove that MLPs do not possess true spatial awareness:
- A fixed random permutation was applied to the pixel coordinates of all images.
- **Result:**
  - Standard Images: **~45–55%** accuracy.
  - Shuffled Images: Drops drastically to **~10–20%** (close to random guessing).
- **Takeaway:** MLPs treat image pixels as independent coordinates rather than extracting translation-invariant localized visual features.

#### 5. Misclassification & Error Analysis
- The MLP struggles heavily when distinguishing visually similar classes with shared background textures or contours (e.g., **Cat vs. Dog**, **Car vs. Truck**, **Deer vs. Horse**).
- Errors predominantly occur on blurry, low-contrast, or small-object images where spatial hierarchies are required to separate subtle distinguishing features.

---

### Part 2: Convolutional Neural Networks (CNN)

#### 1. CNN Architecture
- **Feature Extractor:**
  - `Conv2d(3 -> 32, kernel_size=3, padding=1)` + `ReLU` + `MaxPool2d(2, 2)` (outputs $16 \times 16$)
  - `Conv2d(32 -> 64, kernel_size=3, padding=1)` + `ReLU` + `MaxPool2d(2, 2)` (outputs $8 \times 8$)
- **Classifier:**
  - `Flatten()` $\rightarrow$ `Linear(64 * 8 * 8 -> 128)` + `ReLU` $\rightarrow$ `Linear(128 -> 10)`
- **Optimization:** Adam optimizer ($lr=0.001$) with CrossEntropyLoss.

#### 2. Performance Comparison: CNN vs. MLP

| Metric / Property | Multi-Layer Perceptron (MLP) | Convolutional Neural Network (CNN) |
| :--- | :---: | :---: |
| **Typical Test Accuracy** | ~45% – 55% | **~65% – 75%** |
| **Input Structure** | 1D Flattened Vector (3072) | 2D/3D Tensor ($3 \times 32 \times 32$) |
| **Spatial Locality** | Lost during flattening | Preserved through local receptive fields |
| **Weight Sharing** | No (unique weight per input pair) | Yes (convolutional kernels shared across space) |
| **Parameter Efficiency** | Low (dense FC layers require huge parameter counts) | High (compact kernels extract rich features) |
| **Convergence Speed** | Slower | Faster |

#### 3. Feature Map Visualization
- **Layer 1 Feature Maps:** Capture fine-grained, localized details such as directional edges, color contrasts, and simple textures.
- **Layer 2 Feature Maps:** Synthesize lower-level activations into abstract representations, capturing contours, shapes, and object components.

#### 4. Translation (Shift) Robustness
- Images were vertically shifted using `torch.roll(images, shifts=2, dims=2)`.
- **Result:**
  - **CNN:** Maintains strong classification performance with only a marginal drop in accuracy due to pooling and convolutional shift equivariance.
  - **MLP:** Suffers a substantial accuracy decline because each input neuron is mapped to a fixed pixel index.

#### 5. Ablation Study: Impact of Pooling Layers
To evaluate the contribution of downsampling:
- A pooling-free variant (`CNN_NoPool`) was implemented where MaxPool layers were omitted.
- **Comparison:**

| Architecture Variant | Test Accuracy | Parameter Footprint & Computational Impact |
| :--- | :---: | :--- |
| **Standard CNN (with MaxPool)** | **~65% – 75%** | Compact representation ($64 \times 8 \times 8 = 4,096$ inputs to FC); faster training. |
| **CNN without Pooling (`CNN_NoPool`)** | **~50% – 60%** | Parameter explosion ($64 \times 32 \times 32 = 65,536$ inputs to FC); slow, prone to overfitting. |

- **Conclusion:** Pooling provides critical spatial dimensionality reduction, lowers computation, introduces translation tolerance, and helps prevent overfitting.

---

## 📁 Repository Structure

```text
MLP vs CNN for Image Classification/
│
├── 24110100_Assignment_4.ipynb    # Complete Jupyter Notebook with code, training loops & visualizations
├── README.md                      # Comprehensive project documentation & experimental findings
└── .gitignore                     # Git ignore rules for cached files, datasets, and checkpoints
```

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with `pip`.

### 1. Clone the Repository
```bash
git clone https://github.com/lohithdasari55/MLP-vs-CNN-for-Image-Classification.git
cd MLP-vs-CNN-for-Image-Classification
```

### 2. Install Dependencies
```bash
pip install torch torchvision matplotlib numpy
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook 24110100_Assignment_4.ipynb
```
*(Alternatively, open the notebook in VS Code, JupyterLab, or Google Colab).*

---

## 💡 Acknowledgements & References
- **Dataset:** [CIFAR-10 Dataset by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton](https://www.cs.toronto.edu/~kriz/cifar.html).
- **Framework:** [PyTorch Documentation](https://pytorch.org/docs/stable/index.html).
- Generative AI tools were utilized to support concept clarification and assist in drafting code segments. All code, experiments, and analyses were independently reviewed, executed, and verified.
