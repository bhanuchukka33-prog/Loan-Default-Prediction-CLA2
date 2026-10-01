
import pandas as pd

DATA_PATH = "data/Loan_Default.csv"
TARGET_COLUMN = "Status"

print("=" * 60)
print("LAB 5 - DATA VALIDATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

# Check empty dataset
if df.empty:
    raise ValueError("Dataset is empty!")

# Check duplicate rows
duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)

# Check target column
if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found!"
    )

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check data types
print("\nData Types:")
print(df.dtypes)

print("\n[SUCCESS] Schema Validation Completed!")
