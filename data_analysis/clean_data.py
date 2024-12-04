import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt



# Load the data
file_path = './/data//data_overview.xlsx'  # Replace with your actual file path
data = pd.read_excel(file_path)


# Calculate age
current_year = datetime.now().year
data['Age'] = current_year - data['Year of birth']

# Save the updated data back to Excel
data.to_excel('.//data//data_overview.xlsx', index=False)


# Filter the data based on conditions
cleaned_data = data[
    (data['CT'] != 'no') &  # Keep rows where CT is not 'no'
    (data['Year of birth'].notnull()) &   # Keep rows where Year of birth is not missing
    (data['Age'].notnull()) &             # Keep rows where Age is not missing
    (data['Baby'] != 'yes') &             # Keep rows where Baby is not 'yes'
    (data['Gender'].notnull()) &          # Keep rows where Gender is not missing
    (data['masked_CT'] != 'no')
]

# Save the cleaned data
cleaned_data.to_excel('.//data//data_overview_cleaned.xlsx', index=False)

print(f"Cleaned data saved as: {'.//data//data_overview_cleaned.xlsx'}")



# Filter the data based on conditions
cleaned_data = data[
    (data['CT'] != 'no') &  # Keep rows where CT is not 'no'
    (data['Year of birth'].notnull()) &   # Keep rows where Year of birth is not missing
    (data['Age'].notnull()) &             # Keep rows where Age is not missing
    (data['Baby'] != 'yes') &             # Keep rows where Baby is not 'yes'
    (data['Gender'].notnull()) &
    (data['masked_CT'] != 'no') &
    (data['Classification'].isin(['healthy', 'pathological']))  # Keep rows where Classification is 'healthy' or 'pathological'
]

# Save the cleaned data
cleaned_data.to_excel('.//data//data_overview_binary_cleaned.xlsx', index=False)

print(f"Cleaned data saved as: {'.//data//data_overview_binary_cleaned.xlsx'}")
