#stopped using this method, since when i resized nifti (.nii.gz) files, they became corrupted, and were unusable after

import os
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from skimage.transform import resize

def load_nii(file_path):
    # read neuroimaging file from disk and convert to a numpy array
    # nifti files (.nii.gz) are standard in medical imaging research for storing 3D medical scans
    print(f"Loading file: {file_path}")
    img = nib.load(file_path)
    print(f"File loaded successfully: {file_path}")
    return img.get_fdata()

def save_nii(image, file_path):
    # save processed image back to nifti format
    # using an identity matrix preserves the original image orientation
    print(f"Saving file: {file_path}")
    img = nib.Nifti1Image(image, np.eye(4))  # identity matrix as affine
    nib.save(img, file_path)
    print(f"File saved successfully: {file_path}")

def preprocess_image(image, target_shape=(128, 128, 128)):
    #standardize image dimensions for SWIN model (and to potentially reduce memory usage)
    print(f"Preprocessing image with shape {image.shape}")
    
    # resize using interpolation to maintain quality
    image_resized = resize(image, target_shape, mode='reflect', anti_aliasing=True) #reflect to save items at NIFTI edges
    print(f"Resized image shape: {image_resized.shape}")
    
    return image_resized

def traverse_directories(base_dir):
    # directories in "/Users/adityaranjan/Documents/CuSV/data/BraTS2021_Training_Data" are nested, so need to traverse
    print(f"Traversing directory: {base_dir}")
    nii_files = []
    
    # verify if the directory exists
    if not os.path.exists(base_dir):
        print(f"Error: The base directory {base_dir} does not exist!")
        return nii_files

    print(f"Current working directory: {os.getcwd()}")  # print current working directory
    print(f"Looking for files in {base_dir}...")

    # use os.walk
    for root, dirs, files in os.walk(base_dir):
        print(f"Visiting directory: {root}")
        # check if we have any .nii.gz files
        nii_files_in_dir = [file for file in files if file.lower().endswith('.nii.gz')]
        if nii_files_in_dir:
            print(f"Found .nii.gz files in {root}: {nii_files_in_dir}")
        for file in nii_files_in_dir:
            full_path = os.path.join(root, file)
            nii_files.append(full_path)
    
    return nii_files

def visualize_sample(dataset, num_samples=5): #to visualize images to see if everything processed well
    """
    Visualize a few samples from the dataset.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        image = dataset[i]  # get image
        # show middle slice of the image (since .nii.gz [NIFTI] is a 3D file)
        axes[i].imshow(image[image.shape[0] // 2, :, :], cmap='gray') 
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
    plt.show()

if __name__ == "__main__":
    dataset_dir = "/Users/adityaranjan/Documents/CuSV/data/BraTS2021_Training_Data/"
    processed_dir = "/Users/adityaranjan/Documents/CuSV/processed_data/"
    
    # debug paths
    abs_dataset_dir = os.path.abspath(dataset_dir)
    print(f"Absolute path to dataset directory: {abs_dataset_dir}")
    
    if not os.path.exists(abs_dataset_dir):
        print(f"Error: The directory {abs_dataset_dir} does not exist!")
    else:
        # make output directory if it doesn't exist
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        # find all nifti files in dataset
        file_paths = traverse_directories(abs_dataset_dir)
        
        if not file_paths:
            print(f"No .nii.gz files found in {abs_dataset_dir}")
        else:
            print(f"Found {len(file_paths)} .nii.gz files.")
        
        # process each file
        for idx, file_path in enumerate(file_paths):
            print(f"Processing file {idx+1}/{len(file_paths)}: {file_path}")
            image = load_nii(file_path)
            
            image_resized = preprocess_image(image, target_shape=(128, 128, 128))
            
            # save in 'processed' directory
            file_name = os.path.basename(file_path)
            processed_file_path = os.path.join(processed_dir, file_name)
            save_nii(image_resized, processed_file_path)
        
        # visualize random samples
        random_files = random.sample(file_paths, 5)
        print("\nRandomly selected 5 files for visualization:")
        for file in random_files:
            print(file)
        
        dataset = [load_nii(file) for file in random_files]
        visualize_sample(dataset, num_samples=5)
