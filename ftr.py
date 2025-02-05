import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import glob
import time
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.config import print_config
from monai.transforms import (
    Compose,
    LoadImaged,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    EnsureTyped,
    Activations,
    AsDiscrete,
)
from monai.data import (
    DataLoader,
    Dataset,
    decollate_batch,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.losses import DiceLoss
from monai.utils.enums import MetricReduction

# -------------------------------------------------------------------------
#device setup
def get_mps_device():
    """Try to use Apple's MPS on M1/M2 Mac."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_mps_device()
print_config()
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# data finder
data_dir = "/Users/adityaranjan/Documents/CuSV/data/BraTS2021_Training_Data"
save_dir = "/Users/adityaranjan/Documents/CuSV"

def get_brats_file_paths(root_dir):
    all_dirs = sorted(glob.glob(os.path.join(root_dir, "BraTS2021_*")))
    data_dicts = []
    for pdir in all_dirs:
        flair = glob.glob(os.path.join(pdir, "*_flair.nii.gz"))
        t1ce  = glob.glob(os.path.join(pdir, "*_t1ce.nii.gz"))
        t1    = glob.glob(os.path.join(pdir, "*_t1.nii.gz"))
        t2    = glob.glob(os.path.join(pdir, "*_t2.nii.gz"))
        seg   = glob.glob(os.path.join(pdir, "*_seg.nii.gz"))
        if not (flair and t1ce and t1 and t2 and seg):
            continue
        data_dicts.append({
            "image": [flair[0], t1ce[0], t1[0], t2[0]],
            "label": seg[0]
        })
    return data_dicts

all_data = get_brats_file_paths(data_dir)
print(f"Total Cases Found: {len(all_data)}")

n_total = len(all_data)
n_val   = int(n_total * 0.2)
train_files = all_data[:-n_val]
val_files   = all_data[-n_val:]
print(f"Training set: {len(train_files)} cases, Validation set: {len(val_files)} cases")

# -------------------------------------------------------------------------
# transform
roi_size = (128, 128, 128)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    CropForegroundd(keys=["image", "label"], source_key="image",
                    k_divisible=roi_size, allow_smaller=True),
    RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])

batch_size = 1  # mps limited for 3d volume
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds   = Dataset(data=val_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

print(f"Train Loader: {len(train_loader)} batches, Val Loader: {len(val_loader)} batches")

# -------------------------------------------------------------------------
# fallback model
class SwinUNETRFallback(nn.Module):
    """
    a fallback model uses MONAI SwinTransformer encoder
    encoder returnslist of feature maps; use the last one
    to match ground truth resolution (128³), and apply 1x1x1 conv.
    """
    def __init__(
        self,
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=24,
        spatial_dims=3,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_checkpoint=False,
    ):
        super().__init__()
        self.encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(4, 4, 4),
            patch_size=(4, 4, 4),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )
        # returns list; use last element
        self.out_conv = nn.Conv3d(feature_size * 16, out_channels, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]  # last feature map use
        up_feats = F.interpolate(feats, scale_factor=64, mode="trilinear", align_corners=True)
        logits = self.out_conv(up_feats)
        return logits

# -------------------------------------------------------------------------
# create/define model
model = SwinUNETRFallback(
    img_size=roi_size,
    in_channels=4,
    out_channels=3,
    feature_size=24,  # adjust as needed
    use_checkpoint=False,
).to(device)

# -------------------------------------------------------------------------
# dice loss and inference function
dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(threshold=0.5)

dice_metric = DiceMetric(
    include_background=True,
    reduction=MetricReduction.MEAN_BATCH,
    get_not_nans=True,
)

def inference_sliding_window(batch_data):
    inputs = batch_data["image"].to(device)
    return sliding_window_inference(
        inputs=inputs,
        roi_size=roi_size,
        sw_batch_size=2,
        predictor=model,
        overlap=0.35,
    )

# -------------------------------------------------------------------------
# optimizer/scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# -------------------------------------------------------------------------
# training loop
max_epochs = 5
best_metric = 0.0
best_metric_epoch = -1

for epoch in range(max_epochs):
    print(f"\n=== Epoch [{epoch+1}/{max_epochs}] ===")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = dice_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"  step {step}, train_loss = {loss.item():.4f}")
    epoch_loss /= step
    print(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}")
    scheduler.step()
    # Validation
    model.eval()
    with torch.no_grad():
        dice_metric.reset()
        for val_data in val_loader:
            val_outputs = inference_sliding_window(val_data)
            val_outputs_list = decollate_batch(val_outputs)
            val_labels_list = decollate_batch(val_data["label"].to(device))
            val_pred = [post_pred(post_sigmoid(x)) for x in val_outputs_list]
            dice_metric(y_pred=val_pred, y=val_labels_list)
        mean_dice_val = dice_metric.aggregate().mean().item()
        print(f"Validation Dice: {mean_dice_val:.4f}")
        if mean_dice_val > best_metric:
            best_metric = mean_dice_val
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_swinunetr_m1.pth"))
            print(f"  new best metric: {best_metric:.4f} at epoch {best_metric_epoch} - saved.")
print(f"\nTraining complete! Best validation Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

# -------------------------------------------------------------------------
# inference single case and visualize
test_case = val_files[0]
print("Testing on case:", test_case["image"])

test_ds = Dataset(data=[test_case], transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

model.load_state_dict(torch.load(os.path.join(save_dir, "best_swinunetr_m1.pth")))
model.eval()

with torch.no_grad():
    for batch_data in test_loader:
        outputs = inference_sliding_window(batch_data)
        out_list = decollate_batch(outputs)
        preds = [post_pred(post_sigmoid(x)) for x in out_list]
        seg_3channels = preds[0].cpu().numpy()  # shape: (3, D, H, W)
        slice_idx = seg_3channels.shape[1] // 2
        flair_img_nib = nib.load(test_case["image"][0])
        flair_data = flair_img_nib.get_fdata()
        seg_slice = seg_3channels[:, slice_idx, :, :]
        # channels: [ET=0, WT=1, TC=2]
        pred_wt = seg_slice[1]
        plt.figure("Test Inference", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("FLAIR slice")
        plt.imshow(flair_data[:, :, slice_idx], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("Predicted WT (channel=1)")
        plt.imshow(pred_wt, cmap="jet", alpha=0.5)
        plt.show()
