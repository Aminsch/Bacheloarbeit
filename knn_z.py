import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
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
        data[col] = data[col].map(lambda a: 'missing_values' if pd.isna(a) else le.transform([a])[0])
        label_encoders[col] = le

    data_encoded = pd.get_dummies(data, columns=one_hot_encode_columns, drop_first=True)
    return data_encoded, label_encoders

def knn_imputation(data, impute_columns, n_neighbors=5):
    data_to_impute = data[impute_columns].replace('missing_values', np.nan).astype(float)
    missing_values_before_imputation = data_to_impute.isna().sum()
    print("Fehlende Werte vor der Imputation:")
    print(missing_values_before_imputation)

    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    start_time = time.perf_counter()
    data_imputed = pd.DataFrame(knn_imputer.fit_transform(data_to_impute), columns=impute_columns)
    end_time = time.perf_counter()
    knn_runtime = end_time - start_time

    missing_after = data_imputed.isna().sum()
    filled_values = missing_values_before_imputation - missing_after

    print("Fehlende Werte nach der Imputation:")
    print(missing_after)
    print("Anzahl der gefüllten Felder:")
    print(filled_values)
    print(f'KNN Imputer - Laufzeit: {knn_runtime} Sekunden')

    return data_imputed, missing_values_before_imputation, missing_after, knn_runtime

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
        
        mse[col] = mean_squared_error(original_col, imputed_col)
        mae[col] = mean_absolute_error(original_col, imputed_col)
        
    return mse, mae

data_cleaned = load_and_prepare_data('data/adult/adult.csv', 'data/adult/adult_test.csv')

missing_values_before_encoding = data_cleaned[['workclass', 'occupation', 'native_country']].isna().sum()
print("Fehlende Werte vor der Imputation:")
print(missing_values_before_encoding)

data_encoded, label_encoders = encode_categorical_columns(data_cleaned, ['workclass', 'occupation', 'native_country'], ['race', 'sex', 'education', 'marital_status', 'relationship', 'income'])

missing_values_after_encoding = data_encoded[['workclass', 'occupation', 'native_country']].replace('<NA>', np.nan).isna().sum()
print("Fehlende Werte nach dem Label-Encoding:")
print(missing_values_after_encoding)

train_data, test_data = train_test_split(data_encoded, test_size=0.2, random_state=42)

impute_columns = ['workclass', 'occupation', 'native_country']
train_data_knn_imputed, missing_before_train, missing_after_train, knn_runtime_train = knn_imputation(train_data, impute_columns)
test_data_knn_imputed, missing_before_test, missing_after_test, knn_runtime_test = knn_imputation(test_data, impute_columns)

print("KNN Imputation Ergebnisse:")
print("Fehlende Werte nach der Imputation im Trainingsdatensatz:")
print(train_data_knn_imputed.isna().sum())
print("Fehlende Werte nach der Imputation im Testdatensatz:")
print(test_data_knn_imputed.isna().sum())
print(f'KNN - Laufzeit (Train): {knn_runtime_train} Sekunden')
print(f'KNN - Laufzeit (Test): {knn_runtime_test} Sekunden')

mse_train, mae_train = evaluate_imputation_quality(train_data, train_data_knn_imputed, impute_columns)
print(f"Train MSE: {mse_train}, Train MAE: {mae_train}")

mse_test, mae_test = evaluate_imputation_quality(test_data, test_data_knn_imputed, impute_columns)
print(f"Test MSE: {mse_test}, Test MAE: {mae_test}")
