import pandas as pd
import numpy as np


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
df = pd.read_csv("C:/Users/AMAN/Documents/placement_prediction/dataset/placement_predict_50K_Raw.csv")

data = df.copy()
print("Original Dataset")
print("------------------------")
print(data.head())


print("Dataset Shape:", df.shape)
print("\nData Types:")
print("------------------------")
print(data.dtypes)




print("\nDuplicate Records:", df.duplicated().sum())


for col in data.select_dtypes(include="object").columns:
   data[col] = data[col].str.strip()




print("Missing Values Before Cleaning:")
print(data.isnull().sum())


before_duplicates = data.shape[0]
data = data.drop_duplicates()
after_duplicates = data.shape[0]

print("\nDuplicate Records Removed:",
     before_duplicates - after_duplicates)
num_cols = data.select_dtypes(
   include=np.number
).columns.tolist()


cat_cols = data.select_dtypes(
   exclude=np.number
).columns.tolist()

if len(num_cols) > 0:


   num_imputer = SimpleImputer(
       strategy="mean"
   )


   data[num_cols] = num_imputer.fit_transform(
       data[num_cols]
   )




# ==========================================================
# 5. Fill Missing Categorical Values with Mode
# ==========================================================


if len(cat_cols) > 0:


   cat_imputer = SimpleImputer(
       strategy="most_frequent"
   )


   data[cat_cols] = cat_imputer.fit_transform(
       data[cat_cols]
   )




# ==========================================================
# 6. One-Hot Encoding
# ==========================================================


if len(cat_cols) > 0:


   encoder = OneHotEncoder(
       sparse_output=False,
       handle_unknown="ignore"
   )


   encoded_values = encoder.fit_transform(
       data[cat_cols]
   )




   encoded_df = pd.DataFrame(
       encoded_values,
       columns=encoder.get_feature_names_out(cat_cols)
   )


   encoded_df.reset_index(
       drop=True,
       inplace=True
   )

   numeric_df = data[num_cols].reset_index(
       drop=True
   )

   final_output = pd.concat(
       [
           numeric_df,
           encoded_df
       ],
       axis=1
   )


else:


   final_output = data.copy()


print("\nMissing Values After Cleaning:")
print(final_output.isnull().sum())





final_output.to_csv(
   "C:/Users/AMAN/Documents/placement_prediction/dataset/clean_one_hot_encode_M2.csv",
   index=False
)




print("\n======================================")
print("Original dataset is NOT modified.")
print("Cleaning and One-Hot Encoding completed.")
print("Output file:")
print("clean_one_hot_encoding_M2.csv")
print("======================================")


