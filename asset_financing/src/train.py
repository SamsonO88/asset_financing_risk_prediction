# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, precision_score,\
     roc_curve, roc_auc_score, recall_score


def load_data():
    return pd.read_csv(r"C:\Users\Dell\Documents\my_linux\repos_\asset_financing_risk_prediction\asset_financing\data\processed\clean.csv")
 
 df_clean = load_data()

# check for missing values
df_clean.isna().sum()

# drop missing values because they are less than 5% of the total sample in that specific row
df_clean.dropna(inplace=True)

# compute payment ratio
df_clean["PAYMENT_RATIO"] = round(df_clean["AMT_INCOME_TOTAL"] / df_clean["AMT_CREDIT"], 2)

# Encode the following columns:
# 'CODE_GENDER'
cg_dict = list(df_clean['CODE_GENDER'].unique())
cg_result = {i:v for i,v in enumerate(cg_dict)}
cg_result = {v:i for i,v in cg_result.items()}
df_clean["CODE_GENDER"] = df_clean["CODE_GENDER"].replace(cg_result)
print(cg_result)

# encode 'NAME_INCOME_TYPE'
it_dict = list(df_clean['NAME_INCOME_TYPE'].unique())
it_result = {i:v for i,v in enumerate(it_dict)}
it_result = {v:i for i,v in it_result.items()}
df_clean["NAME_INCOME_TYPE"] = df_clean["NAME_INCOME_TYPE"].replace(it_result)
print(it_result)

# encode 'NAME_EDUCATION_TYPE'
net_dict = list(df_clean['NAME_EDUCATION_TYPE'].unique())
net_result = {i:v for i,v in enumerate(net_dict)}
net_result = {v:i for i,v in net_result.items()}
df_clean["NAME_EDUCATION_TYPE"] = df_clean["NAME_EDUCATION_TYPE"].replace(net_result)
print(net_result)

# encode  'FLAG_OWN_CAR'
foc_dict = list(df_clean['FLAG_OWN_CAR'].unique())
foc_result = {i:v for i,v in enumerate(foc_dict)}
foc_result = {v:i for i,v in foc_result.items()}
df_clean["FLAG_OWN_CAR"] = df_clean["FLAG_OWN_CAR"].replace(foc_result)
print(foc_result)

df_clean["TARGET"] = df_clean["TARGET"].astype(int)
df_clean.info()

# Handle Class Imbalance
# separate majority and minority class
df_majority = df_clean[df_clean["TARGET"] == 0]
df_minority = df_clean[df_clean["TARGET"] == 1]

# Downsample majority class
df_majority_downsampled = resample(
    df_majority,
    replace=False,              # sample without replacement
    n_samples=len(df_minority), # match minority class size
    random_state=42             # reproducibility
)

# combine the minority class with the down sampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Shuffle the dataset
df_downsampled = df_downsampled.sample(frac=0.4, random_state=42).reset_index(drop=True)
print("Original class distribution:\n", df_clean["TARGET"].value_counts())
print("Downsampled class distribution:\n", df_downsampled["TARGET"].value_counts())

# check for duplicates
df_downsampled.duplicated().sum()

# split the dataset for training and testing
X = df_downsampled.drop("TARGET", axis=1)
y = df_downsampled["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
test_pred_ids = X_test["SK_ID_CURR"]
X_train = X_train.drop("SK_ID_CURR", axis = 1)
X_test = X_test.drop("SK_ID_CURR", axis = 1)
X = X.drop("SK_ID_CURR", axis = 1)

## Train and compare at least two models (Random Forest, XGBoost)
# instantiate classifier algorithm
rfc = RandomForestClassifier(n_estimators=200, random_state=42)
xgb = XGBClassifier(n_estimators=200, random_state=42)

# train random forest classifier
rfc.fit(X_train, y_train)

y_pred_rfc = rfc.predict(X_test)

print("Results for Random Forest Classifier")
print("precision score:\n", precision_score(y_test, y_pred_rfc))
print("recall score:\n", recall_score(y_test, y_pred_rfc))
print("roc_auc score:\n", roc_auc_score(y_test, y_pred_rfc))
print(classification_report(y_test, y_pred_rfc))

# train xgboost classifier
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("Results for XGBoost Classifier")
print("precision score:\n", precision_score(y_test, y_pred_xgb))
print("recall score:\n", recall_score(y_test, y_pred_xgb))
print("roc_auc score:\n", roc_auc_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

## standardize and retrain algorithms
# instantiate classifier algorithm
rfc_std = RandomForestClassifier(n_estimators=200, random_state=42)
xgb_std = XGBClassifier(n_estimators=200, random_state=42)

# instantiate standardscaler object
stand = StandardScaler()
# standardize X_train
X_train_std = stand.fit_transform(X_train)
X_test_std = stand.transform(X_test)

# train with standardize data
rfc_std.fit(X_train_std, y_train)

y_pred_rfc_std = rfc_std.predict(X_test_std)
print("Results for Random Forest Classifier after Standardization")
print("precision score:\n", precision_score(y_test, y_pred_rfc_std))
print("recall score:\n", recall_score(y_test, y_pred_rfc_std))
print("roc_auc score:\n", roc_auc_score(y_test, y_pred_rfc_std))
print(classification_report(y_test, y_pred_rfc_std))

# train xgboost classifier based on standardize data
xgb_std.fit(X_train_std, y_train)

y_pred_xgb_std = xgb_std.predict(X_test_std)
print("Results for XGBoost Classifier")
print("precision score:\n", precision_score(y_test, y_pred_xgb_std))
print("recall score:\n", recall_score(y_test, y_pred_xgb_std))
print("roc_auc score:\n", roc_auc_score(y_test, y_pred_xgb_std))
print(classification_report(y_test, y_pred_xgb_std))

## Use cross-validation and show metric performance (ROC AUC, Precision, Recall)
# initiate the algorithm
model_rfc = RandomForestClassifier(n_estimators=300, random_state=42)
model_xgb = XGBClassifier(n_estimators=300, random_state=42)
# instantiate kfold object
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate using cross-validation for random forest
scores = cross_val_score(model_rfc, X, y, cv=skf, scoring="precision")
print(scores)

# Evaluate using cross-validation for xgboost
scores = cross_val_score(model_xgb, X, y, cv=skf, scoring="precision")
print(scores)

# Test predictions
test_preds = xgb.predict_proba(X_test)[:,1]
# Create submission DataFrame
submission = pd.DataFrame({
    "SK_ID_CURR" : test_pred_ids,
    "TARGET" : test_preds
})

# Save submission file
submission.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")

# feature importance for baseline random forest
result_rfc = permutation_importance(rfc, X_test, y_test, n_repeats=10, random_state = 42)
# Display importance
importance_df_rfc = pd.DataFrame({
    'feature': X.columns,\
    'mean_importance': result_rfc.importances_mean,\
    'std_importance': result_rfc.importances_std,\
}).sort_values(by='mean_importance', ascending=False)

print("Feature importnace for baseline Random Forest Algorithm")
print(importance_df_rfc)

plt.barh(importance_df_rfc["feature"], importance_df_rfc["mean_importance"], xerr=importance_df_rfc["std_importance"])
plt.xlabel("Mean Importance (Drop in Score)")
plt.title("Permutation Feature Importance for baseline Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# feature importance for baseline xgboost
result_xgb = permutation_importance(xgb, X_test, y_test, n_repeats=10, random_state = 42)
# Display importance
importance_df_xgb = pd.DataFrame({
    'feature': X.columns,\
    'mean_importance': result_xgb.importances_mean,\
    'std_importance': result_xgb.importances_std,\
}).sort_values(by='mean_importance', ascending=False)

print("Feature importnace for baseline XGBOOST Algorithm")
print(importance_df_xgb)

plt.barh(importance_df_xgb["feature"], importance_df_xgb["mean_importance"], xerr=importance_df_xgb["std_importance"])
plt.xlabel("Mean Importance (Drop in Score)")
plt.title("Permutation Feature Importance for baseline XGBoost")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# feature importance for Random Forest Algorithm using Standardize data
result_rfc_std = permutation_importance(rfc_std, X_test, y_test, n_repeats=10, random_state = 42)
# Display importance
importance_df_rfc_std = pd.DataFrame({
    'feature': X.columns,\
    'mean_importance': result_rfc_std.importances_mean,\
    'std_importance': result_rfc_std.importances_std,\
}).sort_values(by='mean_importance', ascending=False)

print("Feature importance for Random Forest Algorithm using Standardize data")
print(importance_df_rfc_std)

plt.barh(importance_df_rfc_std["feature"], importance_df_rfc_std["mean_importance"], xerr=importance_df_rfc_std["std_importance"])
plt.xlabel("Mean Importance (Drop in Score)")
plt.title("Permutation Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# feature importance for Random Forest Algorithm using Standardize data
result_xgb_std = permutation_importance(xgb_std, X_test, y_test, n_repeats=10, random_state = 42)
# Display importance
importance_df_xgb_std = pd.DataFrame({
    'feature': X.columns,\
    'mean_importance': result_xgb_std.importances_mean,\
    'std_importance': result_xgb_std.importances_std,\
}).sort_values(by='mean_importance', ascending=False)

print("Feature importance for Random Forest Algorithm using Standardize data")
print(importance_df_xgb_std)

plt.barh(importance_df_xgb_std["feature"], importance_df_xgb_std["mean_importance"], xerr=importance_df_xgb_std["std_importance"])
plt.xlabel("Mean Importance (Drop in Score)")
plt.title("Permutation Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()