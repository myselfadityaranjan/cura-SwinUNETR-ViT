#new method drawing directly from dataset, because of corrupted files when cropping to 128x128x128
#this method uses padding to the original 5D tensors of size (1, 1, 240, 240, 155)
#though, very memory intensive and requires lots of swap

import os
import glob
import torch
import gc  # for manual garbage collection
import psutil  # for system memory monitoring
import time  # for tracking memory usage
from tqdm import tqdm
from monai.transforms import ( #still using monai architecture
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
    SpatialPadd,
    OneOf,
    RandRotate90d,
    EnsureChannelFirstd,
)
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.data import Dataset, DataLoader, NibabelReader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
import nibabel as nib

# log memory usage at different stages to monitor and debug memory-related issues
def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"{stage} - Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    
     # if using a GPU, log its memory usage too
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB")

# configuration for directories, device, and training parameters
data_dir = "/Users/adityaranjan/Documents/CuSV/data/BraTS2021_Training_Data"
save_dir = "/Users/adityaranjan/Documents/CuSV"
batch_size = 2
sw_batch_size = 4
max_epochs = 10
infer_overlap = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# retrieves paths for images and corresponding labels, ensuring they match
def get_image_paths_and_data(data_dir):
    print("Traversing Data Directory")
    
    # glob finds files recursively that match specific patterns
    image_files = sorted(glob.glob(os.path.join(data_dir, "**", "*_flair.nii.gz"), recursive=True))
    label_files = sorted(glob.glob(os.path.join(data_dir, "**", "*_seg.nii.gz"), recursive=True))
    
    if not image_files or not label_files:
        raise RuntimeError("No valid image or label files found")
    
    # assert ensures each image has a corresponding label
    assert len(image_files) == len(label_files), "Mismatched number of images and labels"
    
    # create a list of dictionaries pairing images and labels
    data_dicts = []
    for img_path, label_path in zip(image_files, label_files):
        data_dicts.append({
            "image": img_path,
            "label": label_path
        })
    
    return data_dicts

# since original size of images is 240x240x155 (they are not divisible by 32), we need to pad them to be divisible by 32
img_size = (256, 256, 160)

# generates train and validation loaders with transformations
def get_dataloader(data_dicts, batch_size, img_size):
    # split data: 80% for training, 20% for validation
    num_val = max(1, int(len(data_dicts) * 0.2))
    train_files = data_dicts[:-num_val]
    val_files = data_dicts[-num_val:]

    # define transformations
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NibabelReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=img_size, mode="constant"),
        
        # augmentations for normalization/generalization
        OneOf([
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5),
        ]),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
        
        EnsureTyped(keys=["image", "label"]),
    ])
    
    #simple transforms for validation
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NibabelReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=img_size, mode="constant"),
        EnsureTyped(keys=["image", "label"]),
    ])

    # create data loaders for train and validation sets; UPDATED
    train_loader = DataLoader(
        Dataset(data=train_files, transform=train_transforms), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  
        pin_memory=True,  # faster data transfer to GPU
        persistent_workers=False,
        prefetch_factor=2 
    )
    val_loader = DataLoader(
        Dataset(data=val_files, transform=val_transforms), 
        batch_size=1, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2  
    )

    print(f"Data Loaders Ready - Train Size: {len(train_loader.dataset)}, Validation Size: {len(val_loader.dataset)}")
    return train_loader, val_loader

# trains the model for one epoch
def train_epoch(model, loader, loss_function, optimizer):
    model.train()
    epoch_loss = 0
    for batch in tqdm(loader, desc="Training"):
        # MEMORY MANAGEMENT
        torch.cuda.empty_cache() 
        gc.collect() 
        # clear unused memory to prevent GPU out-of-memory errors
        
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad() #reset gradients
        outputs = model(inputs) #forward pass
        loss = loss_function(outputs, labels) #loss calcualte
        loss.backward() #backpropogation
        optimizer.step() #update weights
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)

# validates model and metrics
def validate_epoch(model, loader, dice_metric):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            # MEMORY MANAGEMENT
            torch.cuda.empty_cache() 
            gc.collect() 
            
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # use sliding window inference for large images
            outputs = sliding_window_inference(
                inputs, 
                roi_size=img_size, 
                sw_batch_size=sw_batch_size, 
                predictor=model, 
                overlap=infer_overlap
            )
            
            # process predictions and labels for metrics
            outputs = [AsDiscrete(argmax=True)(i) for i in decollate_batch(outputs)]
            labels = [AsDiscrete(to_onehot=3)(i) for i in decollate_batch(labels)]
            
            dice_metric(y_pred=outputs, y=labels)
        
        metric = dice_metric.aggregate().mean()
        dice_metric.reset()
        return metric

# main training loop
if __name__ == "__main__":
    # MEMORY MANAGEMENT AT START
    torch.cuda.empty_cache()
    gc.collect()
    
    log_memory_usage("Before Model Initialization")

    # initialize model and other training components
    model = SwinUNETR(
        img_size=img_size,
        in_channels=1,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

    log_memory_usage("After Model Initialization")

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH)

    # prepare data and loaders
    data_dicts = get_image_paths_and_data(data_dir)
    train_loader, val_loader = get_dataloader(data_dicts, batch_size, img_size)

    # train for multiple epochs
    best_metric = 0
    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        
        # MEMORY LOGGING
        log_memory_usage(f"Start of Epoch {epoch+1}")
        
        train_loss = train_epoch(model, train_loader, loss_function, optimizer)
        val_metric = validate_epoch(model, val_loader, dice_metric)
        
        scheduler.step() #update lr dynamiclly
        
        print(f"Train Loss: {train_loss:.4f}, Validation Dice: {val_metric:.4f}")
        
        # save the best model based on validation metric
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"New best model saved with Dice: {best_metric:.4f}")
        
        log_memory_usage(f"End of Epoch {epoch+1}")

    print("Training Complete")