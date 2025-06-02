import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from lightly.models.modules import SimCLRProjectionHead

class LIDC_IDRI_SSL_Dataset(Dataset):
    def __init__(self, folder_locations, series_uids, transforms=None, mask_ratio=0.75, patch_size=16, training_phase='mim'):
        super().__init__()
        self.base_folder = r"C:\Users\peter\Masters\Project\processed_data\processed_slices\LIDC"
        self.transforms = transforms
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.series_uids = series_uids

        # Remove series UIDs from folder locations to exclude luna test set
        self.folder_locations = [
            (base, sub) for base, sub in folder_locations
            if sub not in self.series_uids
        ]

        self.slice_paths = []
        for base, sub in self.folder_locations:
            scan_folder = os.path.join(self.base_folder, base, sub)
            png_files = [f for f in os.listdir(scan_folder) if f.endswith(".png")]
            png_files.sort()
            for file in png_files:
                self.slice_paths.append(os.path.join(scan_folder, file))

        # Define default transforms if none are included
        if self.transforms is None:
            self.transforms = {
                "simclr_1": T.Compose([
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.GaussianBlur(3)], p=0.2),
                    T.ToTensor(),
                ]),
                "simclr_2": T.Compose([
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    T.ToTensor(),
                ]),
                "original": T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor()
                ])
            }

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, idx):
        path = self.slice_paths[idx]
        img = Image.open(path).convert("L")

        view1 = self.transforms["simclr_1"](img)
        view2 = self.transforms["simclr_2"](img)
        original = self.transforms["original"](img)

        mask = self._generate_mask(original.shape[1:], self.patch_size, self.mask_ratio)

        return {
            "view1": view1,
            "view2": view2,
            "original": original,
            "mask": mask,
            "target": original,
        }

    def _generate_mask(self, image_shape, patch_size, mask_ratio):
        h, w = image_shape
        assert h % patch_size == 0 and w % patch_size == 0, "Image size must be divisible by patch size"

        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        total_patches = num_patches_h * num_patches_w

        num_masked = int(total_patches * mask_ratio)
        mask_flat = torch.ones(total_patches)
        mask_flat[:num_masked] = 0
        mask_flat = mask_flat[torch.randperm(total_patches)]
        mask_patches = mask_flat.view(num_patches_h, num_patches_w)

        mask_full = mask_patches.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
        return mask_full.unsqueeze(0).float()


class FullModel(nn.Module):
    def __init__(self, backbone, feature_dim=512, projection_dim=128, mim_channels=3, patch_size=16):
        super().__init__()
        self.backbone = backbone

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            backbone_output = self.backbone(dummy_input)
            self.backbone_feature_dim = backbone_output.shape[1]
            print(f"Backbone output feature dimension: {self.backbone_feature_dim}")


        # MIM decoder head
        self.mim_decoder = nn.Sequential(
            nn.Conv2d(self.backbone_feature_dim, 256, 1),  # 1x1 conv for channel reduction
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=224, mode='bilinear', align_corners=False),
            nn.Conv2d(64, mim_channels, 3, padding=1),
            nn.Tanh()
        )

        # Projection head for contrastive learning
        self.projection_head = SimCLRProjectionHead(self.backbone_feature_dim, feature_dim, projection_dim, 2)

        # Training phase
        self.training_phase = 'mim'

    def forward(self, x, mask=None, use_cache = False):
        with torch.cuda.amp.autocast():  # Mixed precision for speed
            features = self.backbone(x)

        if self.training_phase == 'mim':
            # Apply mask at feature level
            if mask is not None:
                # Downsample mask to match feature size
                mask_downsampled = F.adaptive_avg_pool2d(mask, output_size=features.shape[2:])
                masked_features = features * mask_downsampled
            else:
                masked_features = features

            # Decode
            with torch.cuda.amp.autocast():
                mim_output = self.mim_decoder(masked_features)

            return None, mim_output

        else:  # 'cl' phase
            # CL phase: use full network
            with torch.cuda.amp.autocast():
                pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
                projections = self.projection_head(pooled_features)

            return projections, None

    def set_training_phase(self, phase):
        """Set training phase to 'mim' or 'cl'"""
        assert phase in ['mim', 'cl'], "Phase must be 'mim' or 'cl'"
        self.training_phase = phase

        if phase == 'mim':
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.projection_head.parameters():
                param.requires_grad = False
            for param in self.mim_decoder.parameters():
                param.requires_grad = True
        else:  # 'cl' phase
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.mim_decoder.parameters():
                param.requires_grad = False
            for param in self.projection_head.parameters():
                param.requires_grad = True


# Loss functions
def compute_contrastive_loss(proj1, proj2, temp=0.1):
    proj1 = F.normalize(proj1, dim=-1)
    proj2 = F.normalize(proj2, dim=-1)
    sim = torch.mm(proj1, proj2.T) / temp
    labels = torch.arange(proj1.size(0), device=proj1.device)
    contrastive_loss = F.cross_entropy(sim, labels)
    return contrastive_loss

def compute_mim_loss(recon, target):
    return F.mse_loss(recon, target)

# Utilities to get folder locations and series UIDs
def get_folder_locations(lidc_base_folder):
    folder_locations = []
    for base in os.listdir(lidc_base_folder):
        base_path = os.path.join(lidc_base_folder, base)
        if not os.path.isdir(base_path):
            continue
        for sub in os.listdir(base_path):
            sub_path = os.path.join(base_path, sub)
            if not os.path.isdir(sub_path):
                continue
            folder_locations.append((base, sub))
    return folder_locations

def get_series_uids_from_folder(folder_path):
    return set(n.split("_slice_")[0][5:] for n in os.listdir(folder_path))

# Main training function
def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    lidc_base_folder = r"C:\Users\peter\Masters\Project\processed_data\processed_slices\LIDC"
    train_folder = r"C:\Users\peter\Masters\Project\ssl_data\train"
    test_folder = r"C:\Users\peter\Masters\Project\ssl_data\test"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    resume_from_checkpoint = r"checkpoints/checkpoint_epoch_100_cl_full.pth"
    #resume_from_checkpoint = None

    mim_epochs = 75  # Epochs for MIM training
    cl_epochs = 75  # Epochs for CL training
    num_total_epochs = mim_epochs + cl_epochs # Total epochs to train for

    start_epoch = 0  # Will be updated if resuming
    current_phase = 'mim'

    # Get series UIDs
    luna_test_series_ids = get_series_uids_from_folder(test_folder)
    luna_train_series_ids = get_series_uids_from_folder(train_folder)
    print(f"Train series count: {len(luna_train_series_ids)}")
    print(f"Test series count: {len(luna_test_series_ids)}")
    print(f"Overlap count: {len(luna_test_series_ids.intersection(luna_train_series_ids))}")

    # Get folder locations for LIDC dataset
    folder_locations = get_folder_locations(lidc_base_folder)

    # Define transforms
    ssl_transforms = {
        "simclr_1": T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            T.RandomApply([T.RandomRotation(10)], p=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ]),
        "simclr_2": T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.6),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ]),
        "original": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
    }

    # Create dataset and dataloader
    lidc_train = LIDC_IDRI_SSL_Dataset(
        folder_locations=folder_locations,
        series_uids=luna_test_series_ids,
        transforms=ssl_transforms,
        mask_ratio=0.70,
        patch_size=16
    )

    # Load YOLOv5 backbone layers
    from ultralytics import YOLO
    yolo_model = YOLO(r"C:\Users\peter\Masters\Project\yolo\yolov5\yolov5s.pt")
    backbone_layers = list(yolo_model.model.model.children())[:10]
    backbone = nn.Sequential(*backbone_layers)

    # Create model
    model = FullModel(
        backbone=backbone,
        feature_dim=512,
        projection_dim=128,
        mim_channels=3
    )
    model.to(device)

    # Load checkpoint if exists
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Loading checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        current_phase = checkpoint.get('phase', 'mim')
        print(f"Resuming training from epoch {start_epoch}, phase: {current_phase}")

    # Set initial training phase
    model.set_training_phase(current_phase)

    # Create optimizers with different learning rates
    mim_optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 5e-4},
        {'params': model.mim_decoder.parameters(), 'lr': 1e-3}
    ], weight_decay=0.05)

    cl_optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 5e-5}, # Lower rate as fine-tuning MIM backbone
        {'params': model.projection_head.parameters(), 'lr': 1e-3}
    ], weight_decay=0.05)

    # Mixed precision training for speed
    scaler = torch.cuda.amp.GradScaler()

    experiment_name = "runs/sequential_lidc_ssl_experiment_full_backbone"
    if start_epoch > 0:
        print(f"Continuing TensorBoard logs in {experiment_name}")
    writer = SummaryWriter(experiment_name)

    # Training loop
    print(f"Starting sequential training: {mim_epochs} MIM epochs + {cl_epochs} CL epochs")

    for epoch in range(start_epoch, num_total_epochs):
        # Determine current phase
        if epoch < mim_epochs:
            phase = 'mim'
            optimizer = mim_optimizer
            current_batch_size = 16
        else:
            phase = 'cl'
            optimizer = cl_optimizer
            current_batch_size = 16

        # Switch phase if needed
        if phase != model.training_phase:
            print(f"\nSwitching to {phase.upper()} phase at epoch {epoch}")
            model.set_training_phase(phase)

        # Create/update dataloader
        if phase != current_phase or epoch == start_epoch:
            current_phase = phase
            lidc_train_loader = DataLoader(
                lidc_train,
                batch_size=current_batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True  # Keep workers alive between epochs
            )

        model.train()
        total_loss = 0.0

        progress_bar = tqdm(enumerate(lidc_train_loader), total=len(lidc_train_loader),
                            desc=f"Epoch {epoch + 1} ({phase.upper()})")

        for batch_idx, batch in progress_bar:
            optimizer.zero_grad()

            if phase == 'mim':
                # MIM training
                original = batch["original"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)

                # Repeat single channel to 3 channels
                original = original.repeat(1, 3, 1, 1)

                # Forward with mixed precision
                with torch.cuda.amp.autocast():
                    _, mim_output = model(original, mask)
                    loss = compute_mim_loss(mim_output, original)

                # Backward with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            else:  # CL phase
                # Contrastive learning
                view1 = batch["view1"].to(device, non_blocking=True)
                view2 = batch["view2"].to(device, non_blocking=True)

                # Repeat single channel to 3 channels
                view1 = view1.repeat(1, 3, 1, 1)
                view2 = view2.repeat(1, 3, 1, 1)

                # Forward with mixed precision
                with torch.cuda.amp.autocast():
                    proj1, _ = model(view1)
                    proj2, _ = model(view2)
                    loss = compute_contrastive_loss(proj1, proj2, temp=0.1)

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{total_loss / (batch_idx + 1):.4f}"})

        # Log to TensorBoard
        avg_loss = total_loss / len(lidc_train_loader)
        writer.add_scalar(f"Loss/{phase}", avg_loss, epoch + 1)

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}_{phase}_full.pth")
            torch.save({
                'epoch': epoch + 1,
                'phase': phase,
                'model_state_dict': model.state_dict(),
                'mim_optimizer_state_dict': mim_optimizer.state_dict(),
                'cl_optimizer_state_dict': cl_optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    writer.close()
    print("Full training finished.")

if __name__ == "__main__":
    main()
