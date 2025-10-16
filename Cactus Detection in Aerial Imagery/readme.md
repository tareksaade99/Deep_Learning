# Cactus Detection in Aerial Imagery 🌵  
*A Deep Learning Project for Ecological Monitoring*

## Overview
This project focuses on detecting the presence of the cactus species **Neobuxbaumia tetetzo** in aerial imagery. The task supports the broader goals of **autonomous surveillance systems** for protected natural areas, such as those developed under the **VIGIA project in Mexico**, where automated biodiversity monitoring and detection of human impact are essential.

The dataset consists of **17,500 labeled 32×32 images**, each classified as containing a cactus (`1`) or not (`0`). Due to significant **class imbalance** (≈3:1 ratio), careful data handling and augmentation were required to ensure fair and robust model training.

## Preprocessing
- **Data source:** Kaggle competition dataset.  
- **Splits:** Stratified 80/10/10 for train, validation, and test sets.  
- **Processing:** Images loaded as NumPy arrays and converted to PyTorch tensors.  
- **Custom Dataset class:** Ensured compatibility with the PyTorch DataLoader and consistent input formatting `[N, 3, 32, 32]`.

## Handling Class Imbalance
To mitigate class imbalance, the **minority class** (non-cactus images) was **augmented** using horizontal and vertical flips.  
This **data-centric approach** balanced the training set and improved model generalization without relying on class weighting.  
Model performance was evaluated using **F1-score**, a robust metric for imbalanced binary classification.

## Model Architecture
A lightweight **Convolutional Neural Network (CNN)** was implemented in PyTorch:
- 2 convolutional layers with ReLU activations  
- Max-pooling for spatial downsampling  
- Fully connected hidden layer with dropout  
- Final dense layer trained with **BCEWithLogitsLoss**  

The architecture was designed for **efficiency and deployability** in resource-constrained environments.

## Experiments
Training setup:
- **Optimizer:** Adam  
- **Loss function:** BCEWithLogitsLoss  
- **Hyperparameter tuning:** Conducted in two stages  
  - **Architecture tuning:** kernel sizes {3, 5, 7, 9}, dropout rates {0.1–0.7}  
  - **Optimization tuning:** learning rates {0.1–0.0001}, batch sizes {32–128}, weight decay {1e-5}  
- **Best configuration:** kernel size 5, dropout 0.1, learning rate 0.001, batch size 128, weight decay 1e-5  

Training and evaluation were tracked using **Weights & Biases** for reproducibility and monitoring.

## Results
The final model achieved:
- **Validation F1-score:** 0.9852  
- **Validation Accuracy:** 0.978  
- **Training duration:** 40 epochs (early stopping before overfitting)

## Conclusion
This project demonstrates that **compact CNN models** can deliver **high performance** in ecological monitoring tasks with limited resources.  
By addressing **low-resolution inputs**, **class imbalance**, and **efficiency constraints**, the model provides a **practical solution** for real-world environmental surveillance systems.

---
**Keywords:** Deep Learning, CNN, Class Imbalance, Ecological Monitoring, PyTorch, Data Augmentation  

