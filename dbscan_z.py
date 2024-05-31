import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time


def load_and_prepare_data(train_path, test_path):
    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)


    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
                    'marital_status', 'occupation', 'relationship', 'race', 'sex', 
                    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    df_train.columns = column_names
    df_test.columns = column_names

    df = pd.concat([df_train, df_test], ignore_index=True)
    
    relevant_columns = ['workclass', 'occupation', 'native_country']
    df[relevant_columns] = df[relevant_columns].replace(' ?', np.nan)
    
    return df

def encode_categorical_columns(data, label_encode_columns, one_hot_encode_columns):
    label_encoders = {}
    for col in label_encode_columns:
        le = LabelEncoder()
        non_null_data = data[col][data[col].notnull()]
        le.fit(non_null_data)
        data[col] = data[col].map(lambda s: -1 if pd.isna(s) else le.transform([s])[0])
        label_encoders[col] = le

    data_encoded = pd.get_dummies(data, columns=one_hot_encode_columns, drop_first=True)
    return data_encoded, label_encoders


def dbscan_imputation(data, impute_columns, eps=0.5, min_samples=5):
    data_to_impute = data[impute_columns].replace(-1, np.nan).astype(float)
    missing_values_before_imputation = data_to_impute.isna().sum()
    print("Fehlende Werte vor der Imputation:")
    print(missing_values_before_imputation)


    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(imputer.fit_transform(data_to_impute), columns=impute_columns)


    start_time = time.perf_counter()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data_imputed)


    for col in impute_columns:
        for cluster in np.unique(clusters):
            if cluster == -1:
                continue  # Skip noise points
            mask = (clusters == cluster) & data_imputed[col].isna()
            if mask.any():
                fill_value = data_imputed.loc[clusters == cluster, col].median()
                data_imputed.loc[mask, col] = fill_value
    end_time = time.perf_counter()
    dbscan_runtime = end_time - start_time

    missing_after = data_imputed.isna().sum()
    filled_values = missing_values_before_imputation - missing_after

    print("Fehlende Werte nach der Imputation:")
    print(missing_after)
    print("Anzahl der gefüllten Felder:")
    print(filled_values)
    print(f'DBSCAN - Laufzeit: {dbscan_runtime} Sekunden')

    return data_imputed, missing_values_before_imputation, missing_after, dbscan_runtime


def evaluate_imputation_quality(original_data, imputed_data, impute_columns):
    mse = {}
    mae = {}
    
    for col in impute_columns:

        original_col = pd.to_numeric(original_data[col], errors='coerce')
        imputed_col = pd.to_numeric(imputed_data[col], errors='coerce')
        
        if not np.isfinite(original_col).all():
            original_col = np.nan_to_num(original_col, nan=np.nanmean(original_col))
        if not np.isfinite(imputed_col).all():
            imputed_col = np.nan_to_num(imputed_col, nan=np.nanmean(imputed_col))
        
        # Calculate MSE and MAE
        mse[col] = mean_squared_error(original_col, imputed_col)
        mae[col] = mean_absolute_error(original_col, imputed_col)
        
    return mse, mae


data_cleaned = load_and_prepare_data('data/adult/adult.csv', 'data/adult/adult_test.csv')


missing_values_before_encoding = data_cleaned[['workclass', 'occupation', 'native_country']].isna().sum()
print("Fehlende Werte vor der Imputation:")
print(missing_values_before_encoding)

data_encoded, label_encoders = encode_categorical_columns(data_cleaned, ['workclass', 'occupation', 'native_country'], ['race', 'sex', 'education', 'marital_status', 'relationship', 'income'])


missing_values_after_encoding = data_encoded[['workclass', 'occupation', 'native_country']].replace(-1, np.nan).isna().sum()
print("Fehlende Werte nach dem Label-Encoding:")
print(missing_values_after_encoding)


train_data, test_data = train_test_split(data_encoded, test_size=0.2, random_state=42)


impute_columns = ['workclass', 'occupation', 'native_country']
train_data_dbscan_imputed, missing_before_train, missing_after_train, dbscan_runtime_train = dbscan_imputation(train_data, impute_columns)
test_data_dbscan_imputed, missing_before_test, missing_after_test, dbscan_runtime_test = dbscan_imputation(test_data, impute_columns)


print("DBSCAN Imputation Ergebnisse:")
print("Fehlende Werte nach der Imputation im Trainingsdatensatz:")
print(train_data_dbscan_imputed.isna().sum())
print("Fehlende Werte nach der Imputation im Testdatensatz:")
print(test_data_dbscan_imputed.isna().sum())
print(f'DBSCAN - Laufzeit (Train): {dbscan_runtime_train} Sekunden')
print(f'DBSCAN - Laufzeit (Test): {dbscan_runtime_test} Sekunden')


mse_train, mae_train = evaluate_imputation_quality(train_data, train_data_dbscan_imputed, impute_columns)
print(f"Train MSE: {mse_train}, Train MAE: {mae_train}")

mse_test, mae_test = evaluate_imputation_quality(test_data, test_data_dbscan_imputed, impute_columns)
print(f"Test MSE: {mse_test}, Test MAE: {mae_test}")
