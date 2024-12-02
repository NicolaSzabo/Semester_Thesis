import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Load the data
file_path = './/data//data_overview.xlsx'  # Replace with your actual file path
data = pd.read_excel(file_path)

# Display basic info
print("First few rows of the data:")
print(data.head())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Analyze gender distribution
print("\nGender distribution:")
print(data['Gender'].value_counts())

# Analyze classification counts
print("\nClassification counts:")
print(data['Classification'].value_counts())

# Analyze by cause of death
print("\nCause of death counts:")
print(data['Cause of Death'].value_counts())


# Calculate age
current_year = datetime.now().year
data['Age'] = current_year - data['Year of birth']

# Save the updated data back to Excel
data.to_excel('.//data//data_overview.xlsx', index=False)





######### Analyze the Age column
print("\nAge Analysis:")
print(f"Minimum Age: {data['Age'].min()}")
print(f"Maximum Age: {data['Age'].max()}")
print(f"Mean Age: {data['Age'].mean():.2f}")
print(f"Median Age: {data['Age'].median()}")
print(f"Standard Deviation of Age: {data['Age'].std():.2f}")

# Visualize the Age column
plt.hist(data['Age'], bins=10, edgecolor='black', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()





########## Remove rows based on conditions
cleaned_data = data[
    (data['extra Heart iamge'] != 'no') &  # Keep rows where CT is not 'no'
    (data['Year of birth'].notnull()) &   # Keep rows where Year of birth is not missing
    (data['Age'].notnull()) &             # Keep rows where Age is not missing
    (data['Baby'] != 'yes') &             # Keep rows where Baby is not 'yes'
    (data['Gender'].notnull())            # Keep rows where Gender is not missing
]

# Save the cleaned data
cleaned_data.to_excel('.//data//data_overview_cleaned.xlsx', index=False)

print(f"Cleaned data saved as: {'.//data//data_overview_cleaned.xlsx'}")