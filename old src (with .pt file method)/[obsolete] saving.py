#obsolete script now; i incorporated the preprocessing and saving into one script (loading.py), instead of having separate scripts

import os
import torch
from loading import BraTSDataset

def save_preprocessed_data(dataset, output_dir):
    """
    Saves preprocessed data to the specified output directory in .pt format.
    Each file will be named as {original_filename}.pt.
    """
    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving preprocessed data to: {output_dir}")

    for idx in range(len(dataset)):
        # get the file path and image tensor (already preprocessed)
        file_path = dataset.file_paths[idx]
        image_tensor = dataset[idx]

        # do not take '.nii.gz' extension at the end (need .pt)
        file_name = os.path.basename(file_path).replace('.nii.gz', '.pt')
        output_path = os.path.join(output_dir, file_name)

        # save tensor to .pt file
        torch.save(image_tensor, output_path)
        print(f"Saved {file_name} to {output_path}")

def main():
    # paths
    dataset_dir = "/Users/adityaranjan/Documents/CuSV/data/BraTS2021_Training_Data/"
    output_dir = "/Users/adityaranjan/Documents/CuSV/processed_data/"

    # debug
    print(f"Output directory: {os.path.abspath(output_dir)}")

    if not os.path.exists(dataset_dir):
        print(f"Error: Input directory {dataset_dir} does not exist!")
        return

    # load dataset
    dataset = BraTSDataset(dataset_dir)
    if len(dataset) == 0:
        print(f"No files to process in {dataset_dir}")
        return

    # save
    save_preprocessed_data(dataset, output_dir)
    print(f"All files have been preprocessed and saved to {output_dir}")

if __name__ == "__main__":
    main()
