# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# change the directory below to be able to read the data at your end 
dft = pd.read_csv(r"C:\Users\Dell\Documents\my_linux\repos_\asset_financing_risk_prediction\asset_financing\data\raw\application_train.csv")

# general info about dataset
print(dft.shape, dft.info(), dft.duplicated().sum())

list(dft.columns)

# Identify missing values, outliers, and class distribution.
dfta = dft[['SK_ID_CURR','TARGET','CODE_GENDER','FLAG_OWN_CAR','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',\
            'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',\
            'FLAG_PHONE','CNT_FAM_MEMBERS','REG_CITY_NOT_LIVE_CITY',\
            'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT']]
dfta.sample(5)

# check for missing values
dfta.isna().sum()

dfta["AMT_REQ_CREDIT_BUREAU_MON"] = dfta["AMT_REQ_CREDIT_BUREAU_MON"].fillna(dfta["AMT_REQ_CREDIT_BUREAU_MON"].median())
dfta["AMT_REQ_CREDIT_BUREAU_QRT"] = dfta["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(dfta["AMT_REQ_CREDIT_BUREAU_QRT"].median())

# summary of missing values and general information
display(dfta.isna().sum(), dfta.info())

# statistical description for categorical features 
dfta.describe(exclude='number')

# descriptive statistics for numerical features
dfta.describe(exclude='object') 

# outlier detection for:
# Detect Outlier - AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY
def detect_outliers_iqr(df, columns, return_all_outliers=True, flag_outliers=False):
    df_out = df.copy()
    outlier_indices = set()
    outlier_dict = {}

    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Boolean mask for outliers
        mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        # Store outlier indices for this column
        outlier_dict[column] = df[mask]
        if flag_outliers:
            flag_col_name = f'{column}_outlier_flag'
            df_out[flag_col_name] = mask.astype(int)

        if return_all_outliers:
            outlier_indices.update(df[mask].index)
    if return_all_outliers:
        return df_out.loc[list(outlier_indices)]
    elif flag_outliers:
        return df_out
    else:
        return outlier_dict

# detection of outliers
outliers_per_column = detect_outliers_iqr(dfta, ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"], return_all_outliers=False)
print("\nOutliers per column:")
for col, outliers in outliers_per_column.items():
    print(f"\n{col}:\n{outliers}")

# Handle outliers
def replace_outliers_with_median(df, columns):
    df_out = df.copy()
    for column in columns:
        Q1 = df_out[column].quantile(0.25)
        Q3 = df_out[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median = df_out[column].median()

        # Replace outliers with median
        df_out[column] = df_out[column].apply(
            lambda x: median if (x < lower_bound or x > upper_bound) else x
        )
    return df_out

# Replace outliers with median
df_clean = replace_outliers_with_median(dfta, ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"])

# check for duplicate entry
df_clean.duplicated().sum()

# class distribution
df_clean['TARGET'].value_counts()

# 
counts = df_clean['TARGET'].value_counts()
labels = counts.index
sizes = counts.values
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("class distribution of target variable")
plt.legend()
plt.show()

# Summarize dataset structure and key statistics.
# correct the datatype
df_clean["TARGET"] = df_clean["TARGET"].astype('str')
df_clean["REG_CITY_NOT_LIVE_CITY"] = df_clean["REG_CITY_NOT_LIVE_CITY"].astype('boolean')
df_clean["FLAG_OWN_CAR"] = df_clean["FLAG_OWN_CAR"].astype('str')
df_clean["FLAG_PHONE"] = df_clean["FLAG_PHONE"].astype('boolean')

# dataset summary
df_clean.info() 

df_clean.to_csv(r"C:\Users\Dell\Documents\my_linux\repos_\asset_financing_risk_prediction\asset_financing\data\processed\clean.csv", index=False)
