import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Load the data
file_path = './/data//data_overview_binary_cleaned.xlsx'  
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

# Analyze gender distribution for each subclass
gender_distribution_subclass = data.groupby(['Classification', 'Gender']).size().unstack(fill_value=0)





print("\nGender distribution by subclass:")
print(gender_distribution_subclass)

# Analyze classification counts
print("\nClassification counts:")
print(data['Classification'].value_counts())

# Analyze by cause of death
print("\nCause of death counts:")
print(data['Cause of Death'].value_counts())






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
plt.savefig('.//data//data_analysis//age_distribution.png')
plt.show()








