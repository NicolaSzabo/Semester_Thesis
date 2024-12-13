import os
import pandas as pd
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes

def calculate_surface_area(file_path):
    """
    Calculate the surface area (mm²) of a single NIfTI segmentation file.

    Parameters:
        file_path (str): Path to the NIfTI file.

    Returns:
        float: Surface area in square millimeters (mm²), or None if the file doesn't exist.
    """
    if not os.path.exists(file_path):
        return None
    
    nifti_image = nib.load(file_path)
    nifti_data = nifti_image.get_fdata()
    header = nifti_image.header

    # Voxelgröße (Spacing in mm)
    voxel_dims = header.get_zooms()
    
    # Oberfläche berechnen mit Marching Cubes
    verts, faces, _, _ = marching_cubes(nifti_data > 0, level=0.5, spacing=voxel_dims)
    surface_area_mm2 = np.sum(np.linalg.norm(np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                                                     verts[faces[:, 2]] - verts[faces[:, 0]]), axis=1)) / 2
    return surface_area_mm2

def add_surface_to_excel_and_compute_means(excel_path, data_folder, output_excel_path=None):
    """
    Add a 'Surface_mm2' column to the Excel file with calculated surfaces,
    and append the mean surface areas for both classes to the file.

    Parameters:
        excel_path (str): Path to the input Excel file.
        data_folder (str): Folder containing the NIfTI files.
        output_excel_path (str): Path to save the updated Excel file. If None, overwrites the input file.
    """
    # Excel-Daten laden
    excel_data = pd.read_excel(excel_path)
    
    # Filter für 'good' quality Zeilen
    good_quality_data = excel_data[excel_data['quality'] == 'good']

    # Initialisiere die Oberfläche-Spalte
    surfaces = []

    # Oberfläche für jede Datei berechnen
    for _, row in good_quality_data.iterrows():
        patient_id = row['Nr']
        file_path = os.path.join(data_folder, f"{patient_id}.nii.gz")
        surface = calculate_surface_area(file_path)
        surfaces.append(surface if surface is not None else np.nan)

    # Oberfläche in die ursprünglichen Daten einfügen
    excel_data['Surface_mm2'] = np.nan  # Initialisiere die Spalte mit NaN
    excel_data.loc[excel_data['quality'] == 'good', 'Surface_mm2'] = surfaces

    # Mittelwerte für beide Klassen berechnen
    mean_surface_healthy = excel_data.loc[
        (excel_data['quality'] == 'good') & (excel_data['Classification'] == 'healthy'),
        'Surface_mm2'
    ].mean()

    mean_surface_pathological = excel_data.loc[
        (excel_data['quality'] == 'good') & (excel_data['Classification'] == 'pathological'),
        'Surface_mm2'
    ].mean()

    # Ergebnisse hinzufügen
    print(f"Mean Surface for Healthy: {mean_surface_healthy:.2f} mm²")
    print(f"Mean Surface for Pathological: {mean_surface_pathological:.2f} mm²")

    # Ergebnisse in einer neuen Zeile zur Excel-Datei hinzufügen
    summary_row = pd.DataFrame({
        'Nr': ['Summary'],
        'Classification': ['-'],
        'Surface_mm2': ['Mean Healthy: {:.2f} mm², Mean Pathological: {:.2f} mm²'.format(
            mean_surface_healthy, mean_surface_pathological)]
    })

    excel_data = pd.concat([excel_data, summary_row], ignore_index=True)

    # Speichere die aktualisierte Excel-Datei
    if output_excel_path is None:
        output_excel_path = excel_path  # Überschreibt die ursprüngliche Datei

    excel_data.to_excel(output_excel_path, index=False)
    print(f"Updated Excel file saved to: {output_excel_path}")

# Eingabepfade
excel_path = "G://data//data_overview_binary_cleaned_256.xlsx"
data_folder = "G://data_final_without_aorta"
output_excel_path = "G://data//data_overview_binary_cleaned_256.xlsx"

# Starte die Funktion
add_surface_to_excel_and_compute_means(excel_path, data_folder, output_excel_path)
