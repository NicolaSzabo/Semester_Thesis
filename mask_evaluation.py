import numpy as np
import nibabel as nib
import os
import nibabel.processing


def resample_masks():
    
    mask = nib.load('masks/16-1217_5_0_B31S.nii\heart.nii.gz')
    mask_andrea = nib.load('masks_andrea_uncropped/16-1217_label.nii.gz')

    resampled_mask = nibabel.processing.resample_from_to(mask_andrea, mask)
    
    resampled_mask_data = resampled_mask.get_fdata()
    
    nib.save(resampled_mask, 'ground_truth_masks/resampled_mask.nii.gz')
    
    print('Resampling done, new shape: ', resampled_mask_data.shape)
    
    
    
def dice_score(predicted_mask, ground_truth_mask):
    
    """
    Calculate the Dice Similarity Coefficient between two binary masks.
    
    Args:
        predicted_mask (numpy.ndarray): The predicted binary mask.
        ground_truth_mask (numpy.ndarray): The ground truth binary mask.
        
    Returns:
        float: Dice similarity coefficient (DSC) between the two masks.
    """
    
    # Ensure both masks are binary
    predicted_mask = (predicted_mask > 0).astype(np.uint8)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

    # Compute intersection and Dice score
    intersection = np.sum(predicted_mask * ground_truth_mask)
    dice = (2. * intersection) / (np.sum(predicted_mask) + np.sum(ground_truth_mask))
    
    return dice



def calculate_dice_scores(mask_directory, ground_truth_directory):
    
    """
    Calculate Dice scores between predicted masks and ground truth masks.
    
    Args:
        mask_directory (str): Path to directory containing predicted binary masks.
        ground_truth_directory (str): Path to directory containing ground truth masks.
        
    Returns:
        list: A list of Dice scores for each pair of predicted and ground truth masks.
        float: The average Dice score across all segmentations.
    """
    
    dice_scores = []  # List to store Dice scores
    
 
    predicted_mask_path = os.path.join(mask_directory, '16-1217_5_0_B31S.nii/heart.nii.gz')
    ground_truth_mask_path = os.path.join(ground_truth_directory, 'resampled_mask.nii.gz')  
            
            
    # Load the predicted and ground truth masks
    predicted_img = nib.load(predicted_mask_path)
    ground_truth_img = nib.load(ground_truth_mask_path)
            
    predicted_mask_data = predicted_img.get_fdata()
    ground_truth_mask_data = ground_truth_img.get_fdata()
            
    # Calculate Dice score
    dice = dice_score(predicted_mask_data, ground_truth_mask_data)
    dice_scores.append(dice)

    
    # Calculate the average Dice score across all segmentations
    avg_dice = np.mean(dice_scores)
    
    print(dice_scores)
    print(f'Average Dice score across all segmentations: {avg_dice}')
    
    return dice_scores, avg_dice




if __name__ == '__main__':
    
    mask_directory = 'masks/'  # Directory containing predicted segmentation masks
    ground_truth_directory = 'ground_truth_masks/'  # Directory containing ground truth masks
    
    # Calculate Dice scores for all masks
    dice_scores, avg_dice = calculate_dice_scores(mask_directory, ground_truth_directory)
    
    resample_masks()
    