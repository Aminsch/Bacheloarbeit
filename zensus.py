import pandas as pd
import matplotlib.pyplot as plt


# Laden der Datensätze
df_train = pd.read_csv('data/adult/adult.csv', header=None)
df_test = pd.read_csv('data/adult/adult_test.csv', header=None)

# Manuelles Zuweisen von Spaltennamen
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
                'marital_status', 'occupation', 'relationship', 'race', 'sex', 
                'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df_train.columns = column_names
df_test.columns = column_names

# Mergen der beiden DataFrames
df = pd.concat([df_train, df_test], ignore_index=True)

# Zählen der verschiedenen Ergebnisse für die Spalte 'workclass'
unique_workclass_count = df['workclass'].nunique()
print(f"Anzahl verschiedener Ergebnisse für 'workclass': {unique_workclass_count}")

# Anzeigen der verschiedenen Ergebnisse für die Spalte 'workclass'
unique_workclass_values = df['workclass'].unique()
print("Verschiedene Ergebnisse für 'workclass':")
print(unique_workclass_values)

# Häufigkeiten der verschiedenen Ergebnisse für die Spalte 'workclass'
workclass_value_counts = df['workclass'].value_counts()
print("Häufigkeiten der verschiedenen Ergebnisse für 'workclass':")
print(workclass_value_counts)

# Zählen der Anzahl der '?' Werte in der Spalte 'workclass'
question_mark_count = (df['workclass'] == ' ?').sum()
print(f"Anzahl der '?' Werte in der Spalte 'workclass': {question_mark_count}")

# Ersetzen der '?' Werte durch NaN
df['workclass'] = df['workclass'].replace(' ?', pd.NA)

# Berechnen des Prozentsatzes der fehlenden 'workclass' Werte
missing_workclass_percentage = (df['workclass'].isna().mean()) * 100
print(f"Prozentsatz der fehlenden Werte in 'workclass': {missing_workclass_percentage:.2f}%")

workclass_value_counts.plot(kind='bar')
plt.title('Verteilung der Workclass-Werte')
plt.xlabel('Workclass')
plt.ylabel('Häufigkeit')
plt.show()

