import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
import pydicom
import dicom2nifti 
from dicom2nifti import convert_dicom
from dicom2nifti import common
from dicom2nifti import convert_siemens


#Check data for being DICOM or not
file_path = 'THORAX_1_5_B31S_0002/IRM_12-0381.CT.THORAX_IRM_THORAX_ABDOMEN_ROUTINE_(ERWACHSENER).0002.0001.2012.02.22.13.36.05.640625.101668807.IMA'

try:
    dicom_data = pydicom.dcmread(file_path)
    print('This is a valid DICOM file')
except pydicom.errors.InvalidDicomError:
    print('This is not a valid DICOM file')

plt.imshow(dicom_data.pixel_array, cmap='gray')
plt.show()

#Read the DICOM file with pydicom
#print(dicom_data)

#Look at Rescale slope and intercept
print('Rescale Slope: ', dicom_data.RescaleIntercept)
print('Rescale Intercept: ', dicom_data.RescaleSlope)





