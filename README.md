# Feature_Learning_DLGN
# DLGN Interpretability: Code Repository

This repository contains the implementation and analysis code for our research paper:

**"Deep Networks Learn Features From Local Discontinuities in the Label Function"**

## 📌 Overview
This repository provides Jupyter notebooks to run Deep Linearly Gated Networks (DLGN), Variants of DLGN (DLGN-SF, DLGN-VT), and other machine learning models on synthetic and tabular datasets. The best hyperparameters for each model are mentioned in the code comments within the notebooks.

## 📂 Code Structure

### **DLGN and DLGN-SF Experiments**
1. **DLGN_DLGN_SF_Synthetic.ipynb** - Runs DLGN and DLGN-SF models on synthetic datasets.
2. **DLGN_DLGN_SF_Tabular.ipynb** - Runs DLGN and DLGN-SF models on tabular datasets from the paper.

### **Traditional ML Model Experiments**
3. **ML_Synthetic.ipynb** - Runs ML models (Linear/Non-Linear SVM, CART, Random Forest, ReLU Network) on synthetic datasets, selecting the best test accuracy.
4. **ML_Tabular.ipynb** - Runs ML models (Linear/Non-Linear SVM, CART, Random Forest, ReLU Network) on tabular datasets, selecting the best test accuracy.

### **Tree-based Learning Approaches**
5. **Tao_Synthetic_Tabular.ipynb** - Runs the TAO algorithm on synthetic and tabular datasets.
   - Set `DATA_NAME="syn"` for synthetic datasets.
   - Set `DATA_NAME="Tabular"` for tabular datasets.
   - TAO hyperparameters can be adjusted for optimization.
6. **SDT_Synthetic.ipynb** - Runs the SDT algorithm on synthetic datasets.
7. **SDT_Tabular.ipynb** - Runs the SDT algorithm on tabular datasets.

### **Other Decision Tree-Based Models**
8. **ZanDT_Synthetic_Tabular.ipynb** - Runs Zan-DT on synthetic (`"syn"`) and tabular (`"UCI"`) datasets.
9. **Disjunctive_Normal_Networks.ipynb** - Runs DisNN on synthetic (`"syn"`) and tabular (`"UCI"`) datasets.

### **Gated Linear Networks**
10. **Gated_Linear_Networks.ipynb** - Runs GLN on synthetic (`"syn"`) and tabular (`"UCI"`) datasets using [pygln](https://github.com/aiwabdn/pygln.git).

### **DLGN Variants**
11. **DLGN_VT_Tabular.ipynb** - Runs DLGN-VT on tabular datasets.
12. **DLGN_VT_Synthetic.ipynb** - Runs DLGN-VT on synthetic datasets.
13. **DLGN_DT_Synthetic.ipynb** - Runs DLGN-DT on synthetic datasets.

## 🚀 Installation & Usage
### **1. Clone the repository**
```bash
 git clone https://github.com/Doeschate/Feature_Learning_DLGN.git
 cd Feature_Learning_DLGN
```

### **2. Install dependencies**
Ensure you have Python installed. Then, install required dependencies:
```bash
 pip install -r requirements.txt
```

### **3. Running Notebooks**
Open Jupyter Notebook and run the relevant `.ipynb` file cell by cell:
```bash
 jupyter notebook
```
Select the desired notebook and execute each cell sequentially.

### **Example: Running DLGN_DLGN_SF_Synthetic.ipynb**
The notebook contains parameters to modify synthetic data training with different models. Example:
```python
# Change the parameter values to train on different synthetic datasets
input_dim = 20 # Synthetic data input dimension
num_data = 40000 # Total data points
num_levels = 4
threshold = 0 # Data separation distance

optimizer_name = 'Adam'
modep = 'pwc' 
output_dim = 1
num_epoch = 1500
x_epoch = 1500
saved_epochs = list(range(0, 1501, 10))
weight_decay = 0.0
no_of_batches = 10 # Options: [1,10,100]
```

#### **Predefined Best Hyperparameters**
```
For SDI input_dim=20, num_data=40000
DLGN Best Parameters:
Beta     LR  Hidden Layers       Hidden Nodes            Test Accuracy
3      0.020        5           [20, 20, 20, 20, 20]         0.9605

DLGN-SF Best Parameters:
Beta     LR  Hidden Layers        Hidden Nodes           Test Accuracy
30     0.020        4           [10, 10, 10, 10]              0.9743
```

```
For SDII input_dim=100, num_data=60000
DLGN Best Parameters:
Beta     LR  Hidden Layers       Hidden Nodes         Test Accuracy
10     0.010      4             [20, 20, 20, 20]        0.94247

DLGN-SF Best Parameters:
Beta     LR  Hidden Layers        Hidden Nodes          Test Accuracy
3      0.020       4             [10, 10, 10, 10]        0.90293
```

```
For SDIII input_dim=500, num_data=100000
DLGN Best Parameters:
Beta     LR  Hidden Layers       Hidden Nodes         Test Accuracy
10      0.001      3             [10, 10, 10]          0.65036

DLGN-SF Best Parameters:
Beta     LR  Hidden Layers        Hidden Nodes        Test Accuracy
 3     0.010        3             [20, 20, 20]          0.63832
```

## 📎 Citation
If you use this repository in your research, please cite:
```bibtex
@inproceedings{
banerjee2025deep,
title={Deep Networks Learn Features From Local Discontinuities in the Label Function},
author={Prithaj Banerjee and Harish Guruprasad Ramaswamy and Mahesh Lorik Yadav and Chandra Shekar Lakshminarayanan},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=52UtL8uA35}
}
```

## 🔗 References
- [DLGN Paper](https://openreview.net/forum?id=52UtL8uA35)
- [pygln GitHub Repository](https://github.com/aiwabdn/pygln.git)

---

