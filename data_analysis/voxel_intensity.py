import pandas as pd
import numpy as np

def calculate_average_intensities(excel_path):
    """
    Calculate and print the average voxel intensity metrics for each class.

    Parameters:
        excel_path (str): Path to the Excel file containing intensity metrics.
    """
    # Load the Excel file
    excel_data = pd.read_excel(excel_path)

    # Filter rows with 'good' quality only
    good_quality_data = excel_data[excel_data['quality'] == 'good']

    # Group by 'Classification' and compute averages
    mean_intensity_avg = good_quality_data.groupby('Classification')['Mean_Intensity'].mean()
    std_intensity_avg = good_quality_data.groupby('Classification')['Std_Intensity'].mean()

    # Print the results
    print("Average Voxel Intensity Metrics per Class:")
    print("==========================================")
    for classification in mean_intensity_avg.index:
        print(f"Class: {classification}")
        print(f"  - Average Mean Intensity: {mean_intensity_avg[classification]:.2f}")
        print(f"  - Average Intensity Standard Deviation: {std_intensity_avg[classification]:.2f}\n")

# Input path to the Excel file with intensity metrics
excel_path = "G://data//data_overview_binary_cleaned_256.xlsx"

# Run the function
calculate_average_intensities(excel_path)
