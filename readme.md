# CIFAR-100 CNN Kernel Size Comparison Lab

This project explores the impact of convolutional kernel sizes (3x3 vs 5x5) and stride/pooling strategies on image classification performance using Convolutional Neural Networks (CNNs) on the CIFAR-100 dataset. The codebase is implemented in TensorFlow/Keras and includes data preprocessing, model training, evaluation, and result visualization.

---

## **Project Structure**

- `MinhNhat_Mai_Lab06.ipynb` — Main Jupyter notebook with all code, experiments, and analysis.
- `README.md` — Project overview and instructions.
- (Generated) Training history images and plots for both 3x3 and 5x5 kernel experiments.

---

## **Key Features**

- **Data Loading & Preprocessing:**  
  Loads CIFAR-100, normalizes images, and splits into training, validation, and test sets.

- **Data Augmentation:**  
  Applies random horizontal flip and random cropping to improve generalization.

- **Model Architectures:**  
  - **3x3 Kernel Model:** Deeper network with 3x3 convolutions, batch normalization, dropout, and L2 regularization.
  - **5x5 Kernel Model:** Similar structure but with 5x5 convolutions and mixed pooling strategies.

- **Training & Evaluation:**  
  - Uses AdamW optimizer with learning rate scheduling and early stopping.
  - Tracks and plots accuracy/loss curves and learning rate changes.
  - Saves training history and model specs as images for easy comparison.

- **Analysis & Comparison:**  
  - Provides a detailed comparative analysis of 3x3 vs 5x5 kernels.
  - Discusses the impact of kernel size, stride, and pooling on feature extraction and model performance.
  - Includes a summary of challenges faced and recommendations for more complex datasets.

---

## **How to Run**

1. **Install Requirements:**
    ```bash
    pip install tensorflow matplotlib pandas scikit-learn
    ```

2. **Open the Notebook:**
    - Launch JupyterLab or Jupyter Notebook.
    - Open `MinhNhat_Mai_Lab06.ipynb`.

3. **Run All Cells:**
    - Execute each cell in order to reproduce the experiments and visualizations.

---

## **Results**

- **3x3 Kernel Model:**  
  Achieved ~63% test accuracy with strong generalization and minimal overfitting.

- **5x5 Kernel Model:**  
  Achieved ~60% test accuracy, but showed more overfitting and less generalization compared to the 3x3 model.

- **Visualizations:**  
  Training/validation accuracy and loss curves, learning rate schedules, and tabular training history are saved as images for both experiments.

---

## **Analysis Highlights**

- **3x3 kernels** allow for deeper, more expressive networks and better generalization on CIFAR-100.
- **5x5 kernels** extract broader features but may increase overfitting and parameter count.
- **Careful hyperparameter tuning** and sufficient training epochs are crucial for optimal results.
- **Recommendations** for more complex datasets include using deeper/residual architectures, hybrid kernel sizes, advanced augmentation, and transfer learning.

---

## **Credits**

- Developed by Minh Nhat Mai for AIGC5500 Advanced Deep Learning Lab 06.
- Based on TensorFlow/Keras and CIFAR-100 dataset.

---

## **License**

This project is for educational and research purposes.