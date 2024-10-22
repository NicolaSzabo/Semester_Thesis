# This file has 2 functions: dice_score(), calculate_dice_scores().
# The newly created masks from TotalSegmentator are compared with the 'older' masks with the DICE score

import numpy as np
import nibabel as nib
import os


    
def dice_score(mask, ground_truth_mask):
    
    """
    Calculate the Dice Similarity Coefficient between two binary masks.
    
    Args:
        predicted_mask (numpy.ndarray): The predicted binary mask.
        ground_truth_mask (numpy.ndarray): The ground truth binary mask.
        
    Returns:
        float: Dice similarity coefficient (DSC) between the two masks.
    """
    
    # Ensure both masks are binary
    mask = (mask > 0).astype(np.uint8)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

    # Formula for Dice score 
    intersection = np.sum(mask * ground_truth_mask)
    dice = (2. * intersection) / (np.sum(mask) + np.sum(ground_truth_mask))
    
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
    
    # List to store Dice scores
    dice_scores = []  
    
    
    mask_files = [f for f in os.listdir(mask_directory) if f.endswith('.nii.gz')]
    ground_truth_files = [f for f in os.listdir(ground_truth_directory)]
    
                          
    for mask_filename, ground_truth_filename in zip(mask_files, ground_truth_files):
        
        mask_path = os.path.join(mask_directory, mask_filename)
        ground_truth_mask_path = os.path.join(ground_truth_directory, ground_truth_filename)
        
                    
        # Load the predicted and ground truth masks
        predicted_img = nib.load(mask_path)
        ground_truth_img = nib.load(ground_truth_mask_path)

                    
        predicted_mask_data = predicted_img.get_fdata()
        ground_truth_mask_data = ground_truth_img.get_fdata()
            
        # Calculate Dice score
        dice = dice_score(predicted_mask_data, ground_truth_mask_data)
        dice_scores.append(dice)

    
        print(f'Dice score for {mask_filename} and {ground_truth_filename}: {dice}')
                    
                

    # Calculate the average Dice score across all segmentations
    avg_dice = np.mean(dice_scores)
    
    print(f'Average Dice score across all segmentations: {avg_dice}')
    
    return dice_scores, avg_dice




if __name__ == '__main__':
    
    mask_directory = 'masks_heart/'  # Directory containing predicted segmentation masks
    ground_truth_directory = 'ground_truth_masks/'  # Directory containing ground truth masks
    

    # Calculate Dice scores for all masks
    dice_scores, avg_dice = calculate_dice_scores(mask_directory, ground_truth_directory)

    