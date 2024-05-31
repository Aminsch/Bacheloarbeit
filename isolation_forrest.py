import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

def load_and_prepare_data(data_path, admission_type_path, admission_source_path, discharge_disposition_path):
    data = pd.read_csv(data_path, na_values=['?', 'NULL', 'Not Available', 'Unknown/Invalid', 'Not Mapped'])
    admission_type_mapping = pd.read_csv(admission_type_path)
    admission_source_mapping = pd.read_csv(admission_source_path)
    discharge_disposition_mapping = pd.read_csv(discharge_disposition_path)

    admission_type_mapping['admission_type_id'] = admission_type_mapping['admission_type_id'].astype(str)
    discharge_disposition_mapping['discharge_disposition_id'] = discharge_disposition_mapping['discharge_disposition_id'].astype(str)
    admission_source_mapping['admission_source_id'] = admission_source_mapping['admission_source_id'].astype(str)

    data['admission_type_id'] = data['admission_type_id'].astype(str)
    data['discharge_disposition_id'] = data['discharge_disposition_id'].astype(str)
    data['admission_source_id'] = data['admission_source_id'].astype(str)

    data = data.merge(admission_type_mapping, how='left', left_on='admission_type_id', right_on='admission_type_id', suffixes=('', '_admission'))
    data = data.merge(discharge_disposition_mapping, how='left', left_on='discharge_disposition_id', right_on='discharge_disposition_id', suffixes=('', '_discharge'))
    data = data.merge(admission_source_mapping, how='left', left_on='admission_source_id', right_on='admission_source_id', suffixes=('', '_source'))

    data.rename(columns={
        'description': 'admission_type',
        'description_discharge': 'discharge_disposition',
        'description_source': 'admission_source'
    }, inplace=True)

    data.drop(columns=['admission_type_id', 'discharge_disposition_id', 'admission_source_id'], inplace=True)
    data_cleaned = data.drop(columns=['weight', 'max_glu_serum', 'A1Cresult'])

    return data_cleaned

def encode_categorical_columns(data, label_encode_columns, one_hot_encode_columns):
    label_encoder = LabelEncoder()
    for col in label_encode_columns:
        data[col] = label_encoder.fit_transform(data[col].astype(str))

    data_encoded = pd.get_dummies(data, columns=one_hot_encode_columns, drop_first=True)
    return data_encoded

def isolation_forest_imputation(data, impute_columns):
    label_encoders = {}
    for col in impute_columns:
        label_encoder = LabelEncoder()
        mask = data[col].notnull()
        data[col] = data[col].astype(str)
        data.loc[mask, col] = label_encoder.fit_transform(data.loc[mask, col])
        label_encoders[col] = label_encoder

    data_to_impute = data[impute_columns].astype(float)
    missing_values_before_imputation = data_to_impute.isna().sum()
    print("Fehlende Werte vor der Imputation:")
    print(missing_values_before_imputation)

    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(imputer.fit_transform(data_to_impute), columns=impute_columns)

    start_time = time.perf_counter()
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(data_imputed)

    for col in impute_columns:
        for anomaly in np.unique(anomalies):
            if anomaly == -1:
                continue  
            mask = (anomalies == anomaly) & data_imputed[col].isna()
            if mask.any():
                fill_value = data_imputed.loc[anomalies == anomaly, col].median()
                data_imputed.loc[mask, col] = fill_value
    end_time = time.perf_counter()
    iso_forest_runtime = end_time - start_time

    missing_after = data_imputed.isna().sum()
    filled_values = missing_values_before_imputation - missing_after

    print("Anzahl der gefüllten Felder:")
    print(filled_values)
    print(f'Isolation Forest - Laufzeit: {iso_forest_runtime} Sekunden')

    return data_imputed, missing_values_before_imputation, missing_after, iso_forest_runtime


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




data_cleaned = load_and_prepare_data(
    'data/diabetes/diabetic_data.csv',
    'data/diabetes/admission_type_id.csv',
    'data/diabetes/admission_source_id.csv',
    'data/diabetes/discharge_disposition_id.csv'
)

label_encode_columns = ['diag_1', 'diag_2', 'diag_3']
one_hot_encode_columns = ['race', 'gender', 'age', 'change', 'diabetesMed', 'metformin', 'repaglinide', 'nateglinide', 
                          'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 
                          'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 
                          'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                          'metformin-rosiglitazone', 'metformin-pioglitazone']

data_encoded = encode_categorical_columns(data_cleaned, label_encode_columns, one_hot_encode_columns)

missing_values_after = data_encoded[['medical_specialty', 'payer_code', 'admission_type', 'admission_source', 'discharge_disposition']].isna().sum()
print("Fehlende Werte nach dem One-Hot-Encoding:")
print(missing_values_after)

impute_columns = ['medical_specialty', 'payer_code', 'admission_type', 'admission_source', 'discharge_disposition']


train_data, test_data = train_test_split(data_encoded, test_size=0.2, random_state=42)


train_data_iso_imputed, missing_before_train, missing_after_train, iso_runtime_train = isolation_forest_imputation(train_data, impute_columns)
test_data_iso_imputed, missing_before_test, missing_after_test, iso_runtime_test = isolation_forest_imputation(test_data, impute_columns)

train_data_iso_imputed = train_data_iso_imputed.reindex(train_data.index)



print("Isolation Forest Imputation Ergebnisse:")
print("Fehlende Werte nach der Imputation im Trainingsdatensatz:")
print(train_data_iso_imputed.isna().sum())
print("Fehlende Werte nach der Imputation im Testdatensatz:")
print(test_data_iso_imputed.isna().sum())
print(f'Isolation Forest - Laufzeit (Train): {iso_runtime_train} Sekunden')
print(f'Isolation Forest - Laufzeit (Test): {iso_runtime_test} Sekunden')

mse_train, mae_train = evaluate_imputation_quality(train_data, train_data_iso_imputed, impute_columns)
print(f"Train MSE: {mse_train}, Train MAE: {mae_train}")

mse_test, mae_test = evaluate_imputation_quality(test_data, test_data_iso_imputed, impute_columns)
print(f"Test MSE: {mse_test}, Test MAE: {mae_test}")
