#similar to "[update] training.py", but designed to be more memory efficient
#more efficient memory processes and detailed memory logging
#still may be intensive

#same utilities and libraries as before
import os
import glob
import torch
import gc
import psutil
import numpy as np
import time
import logging
from tqdm import tqdm

import nibabel as nib
import torch.nn.functional as F

from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
    SpatialPadd,
    CenterSpatialCropd,
    OneOf,
    RandRotate90d,
    EnsureChannelFirstd,
)
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.data import Dataset, DataLoader, NibabelReader, decollate_batch

# configure logging for both file and console output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)

class BrainTumorSegmentation:
    def __init__(self, data_dir, save_dir, config=None):
        """
        Initialize segmentation training pipeline
        
        Args:
            data_dir (str): Directory containing medical images
            save_dir (str): Directory to save model and logs
            config (dict, optional): Configuration parameters
        """
        # set up a default configuration
        self.config = {
            'batch_size': 1,
            'max_epochs': 10,
            'learning_rate': 1e-4,
            'device': torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
            'infer_overlap': 0.45,
        }
        
        # allow user-provided configurations to override defaults
        if config:
            self.config.update(config)
        
        self.data_dir = data_dir
        self.save_dir = save_dir
        
        # ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
    def log_memory_usage(self, stage):
        """
        log detailed memory usage for monitoring performance
        
        Args:
            stage (str): a label for the current stage of the process
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logging.info(f"{stage} - Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")
        
         # log MPS-specific memory usage if available
        if torch.backends.mps.is_available():
            logging.info(f"MPS Memory Used: {torch.mps.current_allocated_memory() / (1024 * 1024):.2f} MB")
    
    def get_standard_size(self, images, divisor=32):
        """
        calculate a uniform size for all images, ensuring divisibility
        
        Args:
            images (list): list of image file paths
            divisor (int): dimensions will be rounded up to the nearest multiple of this value
        
        Returns:
            tuple: the standardized image dimensions
        """
        sizes = []
        for img_path in images:
            img = nib.load(img_path)
            sizes.append(img.get_fdata().shape[:3])
        
        # Find maximum dimensions
        max_dims = np.max(sizes, axis=0)
        standard_size = tuple(int(np.ceil(dim / divisor) * divisor) for dim in max_dims)
        
        logging.info(f"Standard image size: {standard_size}")
        return standard_size
    
    def prepare_data(self):
        """
       prepare image and label paths and compute uniform dimensions
        
        Returns:
            tuple: training data, validation data, and standard image size
        """
        logging.info("Preparing medical image data")
        
        # search for all FLAIR and segmentation label files
        image_files = sorted(glob.glob(os.path.join(self.data_dir, "**", "*_flair.nii.gz"), recursive=True))
        label_files = sorted(glob.glob(os.path.join(self.data_dir, "**", "*_seg.nii.gz"), recursive=True))
        
        # check that files exist and are correctly paired
        if not image_files or not label_files:
            raise RuntimeError("No valid image or label files found")
        
        assert len(image_files) == len(label_files), "Mismatched number of images and labels"
        
        # limit the dataset size for testing purposes; CAN CHANGE
        image_files = image_files[:3]
        label_files = label_files[:3]
        
        # calculate a uniform size for the images
        standard_size = self.get_standard_size(image_files)
        
        data_dicts = [
            {"image": img_path, "label": label_path}
            for img_path, label_path in zip(image_files, label_files)
        ]
        
        # split data into training and validation sets
        num_val = max(1, int(len(data_dicts) * 0.2))
        train_files = data_dicts[:-num_val]
        val_files = data_dicts[-num_val:]
        
        return train_files, val_files, standard_size
    
    def create_transforms(self, standard_size):
        """
         define data preprocessing and augmentation pipelines
        
        Args:
            standard_size (tuple): uniform image size for padding and cropping
        
        Returns:
            tuple: training and validation transformation pipelines
        """
        train_transforms = Compose([
            LoadImaged(keys=["image", "label"], reader=NibabelReader()),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            
            # pad and crop to ensure uniform dimensions
            SpatialPadd(keys=["image", "label"], spatial_size=standard_size, mode="constant"),
            CenterSpatialCropd(keys=["image", "label"], roi_size=standard_size),
            
            # apply random augmentations for training
            OneOf([
                RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
                RandRotate90d(keys=["image", "label"], prob=0.3),
            ]),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            
            EnsureTyped(keys=["image", "label"]),
        ])
        
        # minimal preprocessing for validation
        val_transforms = Compose([
            LoadImaged(keys=["image", "label"], reader=NibabelReader()),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            SpatialPadd(keys=["image", "label"], spatial_size=standard_size, mode="constant"),
            CenterSpatialCropd(keys=["image", "label"], roi_size=standard_size),
            EnsureTyped(keys=["image", "label"]),
        ])
        
        return train_transforms, val_transforms
    
    def create_dataloaders(self, train_files, val_files, train_transforms, val_transforms):
        """
        create data loaders for efficient batch processing
        
        Returns:
            tuple: training and validation data loaders
        """
        train_loader = DataLoader(
            Dataset(data=train_files, transform=train_transforms), 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )
        val_loader = DataLoader(
            Dataset(data=val_files, transform=val_transforms), 
            batch_size=1, 
            shuffle=False, 
            num_workers=0,
            pin_memory=False
        )
        
        logging.info(f"Data Loaders Ready - Train Size: {len(train_loader.dataset)}, Validation Size: {len(val_loader.dataset)}")
        return train_loader, val_loader
    
    def train(self):
        """main training pipeline, including setup and loops"""
        try:
            # prepare data and transformations
            train_files, val_files, standard_size = self.prepare_data()
            train_transforms, val_transforms = self.create_transforms(standard_size)
            
            # create data loaders
            train_loader, val_loader = self.create_dataloaders(
                train_files, val_files, train_transforms, val_transforms
            )
            
             # initialize the SwinUNETR model
            model = SwinUNETR(
                img_size=standard_size,
                in_channels=1,
                out_channels=3,
                feature_size=24,
                use_checkpoint=False,
            ).to(self.config['device'])
            
            # define the loss, optimizer, and metrics
            loss_function = DiceLoss(to_onehot_y=True, softmax=True)
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.config['learning_rate'], 
                weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
            dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH)
            
            # initialize metric tracking
            best_metric = 0
            for epoch in range(self.config['max_epochs']):
                logging.info(f"Epoch {epoch+1}/{self.config['max_epochs']}")
                
                # training
                model.train()
                train_loss = self._train_epoch(model, train_loader, loss_function, optimizer)
                
                # validation
                val_metric = self._validate_epoch(model, val_loader, dice_metric)
                
                # learning rate step
                scheduler.step()
                
                # model saving
                if val_metric > best_metric:
                    best_metric = val_metric
                    save_path = os.path.join(self.save_dir, "best_model.pth")
                    torch.save(model.state_dict(), save_path)
                    logging.info(f"New best model saved with Dice: {best_metric:.4f}")
        
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise
    
    def _train_epoch(self, model, loader, loss_function, optimizer):
        """execute one training epoch
        
        Returns:
            float: average training loss"""
        epoch_loss = 0
        for batch in tqdm(loader, desc="Training"):
            # load batch data to device
            inputs = batch["image"].to(self.config['device'])
            labels = batch["label"].to(self.config['device'])
            
            # zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            # forward pass, compute loss, and backpropagate
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            #bckwrd
            loss.backward()
            optimizer.step()
            
            # accumulate loss
            epoch_loss += loss.item()
        
        return epoch_loss / len(loader)
    
    def _validate_epoch(self, model, loader, dice_metric):
        """ evaluate model performance on the validation set
        
        Returns:
            float: average validation metric"""
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):
                inputs = batch["image"].to(self.config['device'])
                labels = batch["label"].to(self.config['device'])
                
                # sliding window inference
                outputs = F.sliding_window_inference(
                    inputs, 
                    window_size=inputs.shape[2:], 
                    overlap=self.config['infer_overlap'], 
                    predictor=model
                )
                
                # post-processing
                outputs = [torch.argmax(i, dim=0) for i in decollate_batch(outputs)]
                labels = [torch.argmax(i, dim=0) for i in decollate_batch(labels)]
                
                dice_metric(y_pred=outputs, y=labels)
            
            metric = dice_metric.aggregate().mean()
            dice_metric.reset()
            return metric

def main():
    data_dir = "/Users/adityaranjan/Documents/CuSV/data/BraTS2021_Training_Data"
    save_dir = "/Users/adityaranjan/Documents/CuSV"
    
    # optional custom configs
    config = {
        'batch_size': 1,
        'max_epochs': 10,
        'learning_rate': 1e-4
    }
    
    # run training and optimize
    segmentation = BrainTumorSegmentation(data_dir, save_dir, config)
    segmentation.train()

if __name__ == "__main__":
    main()