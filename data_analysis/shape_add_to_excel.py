import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def update_and_calculate_compactness(excel_path):
    """
    Update compactness values in the Excel file using normalized compactness formula
    and calculate average compactness for 'healthy' and 'pathological' classes.

    Parameters:
        excel_path (str): Path to the Excel file.

    Returns:
        dict: Average compactness and standard deviations per class.
    """
    # Load the Excel file
    excel_data = pd.read_excel(excel_path)

    # Update compactness using the normalized formula: Surface / Volume^(2/3)
    excel_data['Compactness'] = excel_data.apply(
        lambda row: row['Surface_mm2'] / (row['Volume_mL'] ** (2 / 3)) if row['Volume_mL'] > 0 else np.nan, axis=1
    )

    # Save the updated Excel file
    excel_data.to_excel(excel_path, index=False)
    print(f"Updated compactness values saved to: {excel_path}")

    # Filter the data for 'good' quality scans
    good_quality_data = excel_data[excel_data['quality'] == 'good']

    # Initialize lists for compactness values
    healthy_compactness = []
    pathological_compactness = []

    # Iterate over rows to separate compactness by class
    for _, row in good_quality_data.iterrows():
        classification = row['Classification']
        compactness = row['Compactness']

        if classification == 'healthy' and not np.isnan(compactness):
            healthy_compactness.append(compactness)
        elif classification == 'pathological' and not np.isnan(compactness):
            pathological_compactness.append(compactness)

    # Calculate averages and standard deviations
    avg_healthy_compactness = np.mean(healthy_compactness) if healthy_compactness else 0
    std_healthy_compactness = np.std(healthy_compactness) if healthy_compactness else 0
    avg_pathological_compactness = np.mean(pathological_compactness) if pathological_compactness else 0
    std_pathological_compactness = np.std(pathological_compactness) if pathological_compactness else 0

    # Print results
    print(f"Average Healthy Compactness: {avg_healthy_compactness:.4f} (± {std_healthy_compactness:.4f})")
    print(f"Average Pathological Compactness: {avg_pathological_compactness:.4f} (± {std_pathological_compactness:.4f})")

    # Improved Boxplot
    plt.figure(figsize=(8, 6))
    boxprops = dict(color='darkblue', linewidth=2)
    medianprops = dict(color='darkblue', linewidth=2)
    whiskerprops = dict(color='darkblue', linewidth=1.5)
    capprops = dict(color='darkblue', linewidth=2)

    plt.boxplot([healthy_compactness, pathological_compactness],
                labels=['Healthy', 'Pathological'],
                boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops, capprops=capprops, patch_artist=True,
                widths=0.6)

    plt.title('Compactness Comparison', fontsize=16)
    plt.ylabel('Compactness', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig('compactness_boxplot.png', dpi=300)
    print("Boxplot saved as 'compactness_boxplot.png' in the current directory.")

    return {
        "Healthy": {"Mean": avg_healthy_compactness, "Std": std_healthy_compactness},
        "Pathological": {"Mean": avg_pathological_compactness, "Std": std_pathological_compactness}
    }


# Input path
excel_path = "/home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview_binary_cleaned_256.xlsx"

# Run the function
average_compactness = update_and_calculate_compactness(excel_path)
