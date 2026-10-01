import pandas as pd

df = pd.read_csv("data/Loan_Default.csv")

print("=" * 50)
print("LOAN DEFAULT DATASET")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())