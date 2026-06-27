import pandas as pd

# Load your CSV
df = pd.read_csv('03_FIDE_Chess_Ratings.csv')

# Print all column names
print("Columns in your CSV:")
for col in df.columns:
    print(f"  - {col}")

print("\nFirst 5 rows:")
print(df.head())

print(f"\nTotal rows: {len(df)}")
