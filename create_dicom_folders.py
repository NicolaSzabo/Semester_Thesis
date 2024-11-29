import pandas as pd
import os

def create_folders_from_excel(file_path, column_name, output_directory):
    """
    Creates folders based on the values in a specified column of an Excel file.
    
    Parameters:
    - file_path (str): Path to the Excel file.
    - column_name (str): The name of the column containing folder names.
    - output_directory (str): Path where folders will be created.
    """
    # Load the Excel file
    data = pd.read_excel(file_path)
    print(data.columns)  # Print the column names
    
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Loop through the values in the specified column and create folders
    for folder_name in data[column_name]:
        folder_path = os.path.join(output_directory, str(folder_name))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")

# Example usage
if __name__ == "__main__":
    # Path to your Excel file
    file_path = "C://Users//nicol//OneDrive//Desktop//semester_thesis//Project//data_overview.xlsx"
    # Column name containing the folder names (e.g., "Nr" from your table)
    column_name = "Nr"
    # Path where the folders should be created
    output_directory = "G://create_nifti_files"
    
    create_folders_from_excel(file_path, column_name, output_directory)
