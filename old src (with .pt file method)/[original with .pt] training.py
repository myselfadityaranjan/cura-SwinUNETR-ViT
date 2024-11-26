#this script is obsolete, since it used the old method of .pt files (that got corrupted when resizing)

import os
import glob
import torch
from tqdm import tqdm #progress bar
from monai.transforms import ( #using monai SWINUnetR architecture
    Compose,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
    Activations,
    AsDiscrete,
)
from monai.losses import DiceLoss #diceloss loss calculator
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference

# debugging utility
def announce(stage, details=""):
    print(f"--- {stage} ---")
    if details:
        print(details)

# utility class for tracking averages of metrics (e.g., loss, dice score)
class AverageMeter:
    def __init__(self):
        self.reset()

    # reset internal state to start fresh
    def reset(self):
        self.sum = 0
        self.count = 0

    # update running total and count
    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    # calculate the current average
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0

# configuration settings for paths, device, and parameters
data_dir = "/Users/adityaranjan/Documents/CuSV/processed_data"
save_dir = "/Users/adityaranjan/Documents/CuSV"
roi = (128, 128, 128)  # define 3D patch size
batch_size = 2
sw_batch_size = 4
max_epochs = 100
infer_overlap = 0.5 #overlap ratio for SWIN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# disabling CuDNN for conssitent memory usage
torch.backends.cudnn.benchmark = False

# debugging initialization
announce("Setup", f"Device: {device}, Data Directory: {data_dir}")

# file searching in processed data directory
def get_file_paths(data_dir):
    announce("Traversing Processed Data Directory", f"Scanning files in: {data_dir}")
    data_dicts = []

    # have a 'flair' and 'seg' for image and segmentation tag
    image_files = sorted(glob.glob(os.path.join(data_dir, "*_flair.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(data_dir, "*_seg.nii.gz"))) #can incorporate other modalities later as well

    # match based on file name
    for image_path in image_files:
        identifier = os.path.basename(image_path).split("_")[0]  # i.e. "BraTS2021_00000"
        matching_label = next(
            (label for label in label_files if identifier in os.path.basename(label)), None
        )
        if matching_label:
            data_dicts.append({"image": image_path, "label": matching_label})

    if not data_dicts:
        raise RuntimeError(f"No valid image-label pairs found in directory: {data_dir}")

    announce("Files Found", f"Found {len(data_dicts)} image-label pairs.")
    return data_dicts

# create dataloaders
def get_dataloader(data_dicts, roi, batch_size):
    num_val = max(1, int(len(data_dicts) * 0.2))
    train_files = data_dicts[:-num_val]
    val_files = data_dicts[-num_val:]

    #define transformations for augmentation and normalization
    train_transforms = Compose([
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
        EnsureTyped(keys=["image", "label"]),
    ])
    val_transforms = Compose([
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    #create ds and loaders
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)

    #log details abt datasets
    announce("Data Loaders Ready", f"Train Size: {len(train_loader.dataset)}, Validation Size: {len(val_loader.dataset)}")
    return train_loader, val_loader

data_dicts = get_file_paths(data_dir)
train_loader, val_loader = get_dataloader(data_dicts, roi, batch_size)

# initialize the model, loss function, optimizer, and metric
model = SwinUNETR(
    img_size=roi,
    in_channels=1,
    out_channels=3,
    feature_size=48,
    use_checkpoint=True,
).to(device)

# dice loss is used as the primary metric for segmentation performance
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH)

# function to handle the training process for one epoch
def train_epoch(model, loader):
    model.train()
    epoch_loss = AverageMeter()
    progress = tqdm(loader, desc="Training", unit="batch")
    for batch in progress:
        try:
            # transfer data to device and perform forward/backward pass
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # update loss tracking and log progress
            epoch_loss.update(loss.item(), inputs.size(0))
            progress.set_postfix(loss=epoch_loss.avg)
        except RuntimeError as e:
            print(f"Runtime error during training: {e}")
            continue
    return epoch_loss.avg

# function to validate model performance using dice metric
def validate_epoch(model, loader):
    model.eval()
    dice_values = AverageMeter()
    progress = tqdm(loader, desc="Validation", unit="batch")
    with torch.no_grad():
        for batch in progress:
            try:
                inputs, labels = batch["image"].to(device), batch["label"].to(device)
                outputs = sliding_window_inference(inputs, roi, sw_batch_size, model, overlap=infer_overlap)

                 # post-processing for dice calculation
                post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
                outputs = [post_trans(i) for i in decollate_batch(outputs)]
                dice_metric(y_pred=outputs, y=decollate_batch(labels))
            except RuntimeError as e:
                print(f"Runtime error during validation: {e}")
                continue

        #aggregate dice scores across dataset
        dice_values.update(dice_metric.aggregate().item(), len(loader))
        dice_metric.reset()
    return dice_values.avg

# main training loop
if __name__ == "__main__":
    best_dice = 0.0
    for epoch in range(max_epochs):
        announce(f"Epoch {epoch + 1}/{max_epochs}")
        try:
            train_loss = train_epoch(model, train_loader)
            val_dice = validate_epoch(model, val_loader)
            scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Validation Dice: {val_dice:.4f}")

            if val_dice > best_dice:
                best_dice = val_dice
                model_path = os.path.join(save_dir, "best_model.pth")
                torch.save(model.state_dict(), model_path)
                announce("Model Saved", f"New best model saved at {model_path}")
        except RuntimeError as e:
            print(f"Error during Epoch {epoch + 1}: {e}")
            break

    announce("Training Complete")
