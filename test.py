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


