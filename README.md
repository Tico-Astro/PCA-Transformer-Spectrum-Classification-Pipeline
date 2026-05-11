# (PCA-Transformer Spectra Classification Pipeline)PTSCP
A method that first reduces the dimension of the original spectra data using PCA and then classifies it using the Transformer model.

## Current Status & Data Availability

This project is currently undergoing further refinement for the future work.  As a result, the pre-trained model weights and the integrated training datasets are not publicly available at this time.  We plan to release more assets in the future.  Thank you for your understanding.

## Acknowledgements

This project is built upon [Mgformer](https://doi.org/10.1016/j.engappai.2024.108633). We gratefully acknowledge their contributions.

The source code or model has been adapted from the following repository:
(https://github.com/LiYunxiaoboy/mgformer)

## 📂 Project Structure

This section outlines the functions of the key directories and files in this repository.

### Directories

| Folder | Description |
| :--- | :--- |
| **`PCA_evaluate/`** | Stores PCA vectors extracted from the training set, used specifically during the model evaluation phase. |
| **`PCA_run/`** | Contains PCA vectors optimized for large-scale classification tasks on real-world datasets. |
| **`data/`** | Includes `Training_Set_7P_PCA_SNmod_mock_train`, the integrated training set for performance benchmarking. (Not availabe now)|
| **`demo_data/`** | A sample dataset used to demonstrate the classification pipeline and sample outputs. |
| **`moudel/`** | **Core Source Code**: Contains the model architecture and implementation of **Mgformer**. |

---

### Scripts & Jupyter Notebooks

| File | Category | Description |
| :--- | :--- | :--- |
| **`Model_Train-Spectrum-for_demo_1.ipynb`** | Training | Main pipeline for Transformer model training and performance evaluation. (Some deletions)|
| **`Std_traning_set_code.ipynb`** | Preprocessing | Scripts for spectral stacking, PCA dimensionality reduction, and dataset generation. (Some deletions)|
| **`PCA_host_substract.py`** | Preprocessing | An integrated program for host-galaxy subtraction and PCA vector extraction. |
| **`Sdss_pipeline_simp.ipynb`** | Inference | Demonstration of spectral preprocessing and sample classification results. (Some deletions) |
| **`utils.py`** | Helpers | Utility functions for data loading, augmentation, and training support. |

---

### Pre-trained Models (.pth)

| Model File | Use Case |
| :--- | :--- |
| **`Test_model_sp106-evaluation.pth`** | Checkpoint used for reproducing model evaluation metrics. (Not availabe now) |
| **`test_model_SP106-use4.pth`** | **Production Model**: Optimized for actual scientific inference on large datasets. (Not availabe now)  |

---


