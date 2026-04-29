## Asset Financing Risk Prediction
This project aims to predict if an asset should be financed or not.
### Problem Statement
Using customer historical blue print, we predict if a customer should be financed or not.
### Objectives
1. Find the summary statistics of the variable charges in the dataset
2. Display a table that contains the number of people in each region
3. Visualize the relationship among all features using a scatterplot matrix
4. Train a model on the data
5. Evaluate the model performance
6. Improve the model performance by Adding nonlinear relationship 
### Project Structure
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── notebooks/
    ├── src/
    │   ├── data_preprocessing.py
    │   ├── train.py
    │   └── evaluate.py
    ├── models/
    ├── reports/
    │   └── figures/
    ├── requirements.txt
    ├── README.md
    └── .gitignore

Installation
### Clone repository
git clone git@github.com:SamsonO88/asset_financing_risk_prediction.git

### Navigate into project
cd asset_financing_risk_prediction

### Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Evaluation Metrics
Metrics used: 
    1) F1 Score
    2) ROC-AUC
### Technologies Used
    1) Python
    2) Pandas, NumPy
    3) Scikit-learn
    4) Matplotlib / Seaborn
    5) Jupyter Notebook
### License
    MIT License
