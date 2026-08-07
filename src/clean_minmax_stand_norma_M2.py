import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import matplotlib.pyplot as plt

file_path = "C:/Users/AMAN/Documents/placement_prediction/dataset/placement_predict_50K_Raw.csv"
df = pd.read_csv(file_path)
print("Original Dataset")

print("------------------------")
print(df.head())
print("Dataset Shape:", df.shape)
print("\nData Types:")
print("------------------------")
print(df.dtypes)


print("\nMissing Values:")
print("------------------------")
print(df.isnull().sum())


print("\nDuplicate Records:", df.duplicated().sum())


df = df.drop_duplicates()

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].mean())

categorical_columns = df.select_dtypes(include=['object']).columns

for column in categorical_columns:
   df[column] = df[column].fillna(df[column].mode()[0])

for  column in categorical_columns:
   df[column] = df[column].str.strip()


numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

print("\nNumeric Columns:")
print(list(numeric_columns))

standard_scaler = StandardScaler()
standardized = standard_scaler.fit_transform(df[numeric_columns])


for i, col in enumerate(numeric_columns):
   df[col + "_Standardized"] = standardized[:, i]

minmax_scaler = MinMaxScaler()
scaled = minmax_scaler.fit_transform(df[numeric_columns])
for i, col in enumerate(numeric_columns):
   df[col + "_Scaled"] = scaled[:, i]

normalizer = Normalizer(norm='l2')
normalized = normalizer.fit_transform(df[numeric_columns])
for i, col in enumerate(numeric_columns):
   df[col + "_Normalized"] = normalized[:, i]

print("\n Display Results after Preprocessed Dataset")
print(df.head())


print("\nDataset Shape:", df.shape)


print("\nDataset Information")
print(df.info())


print("\nColumns in Dataset:")
print(df.columns)


print("\nMissing Values After Preprocessing")
print(df.isnull().sum())


print("\nDuplicate Records After Preprocessing")
print(df.duplicated().sum())

df.to_csv("C:/Users/AMAN/Documents/placement_prediction/dataset/clean_minmax_stand_norma_M2.csv", index=False)
print("\n Preprocessed dataset saved successfully")
pf=pd.read_csv("C:/Users/AMAN/Documents/placement_prediction/dataset/clean_minmax_stand_norma_M2.csv")

pf.hist(figsize=(12, 10), bins=10, edgecolor='black')


plt.suptitle("Histogram of Preprocessed Placement Dataset")
plt.tight_layout()
plt.show()

