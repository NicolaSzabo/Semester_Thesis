import pandas as pd
import numpy as np

# 1. Load the Excel file
excel_path = '/home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview_binary_cleaned_256.xlsx'
data = pd.read_excel(excel_path)

# 2. Compute Compactness using Volume and Surface Area
# Compactness formula: Surface^3 / Volume^2
data['Compactness'] = (data['Surface_mm2'] ** 3) / (data['Volume_mL'] ** 2)

# 3. Check for NaN or Infinite values (due to division errors)
data['Compactness'] = data['Compactness'].replace([np.inf, -np.inf], np.nan).fillna(0)

# 4. Save the updated file with the new column
output_path = '/home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview_binary_cleaned_256.xlsx'
data.to_excel(output_path, index=False)

print("Compactness feature added successfully and saved to:", output_path)
