# Paired-Sampling Contrastive Framework Configuration
# https://arxiv.org/abs/2508.14980

# Data paths (adjust these for your setup)
paths = dict(
    train_csv="/dataset/train_matched_09_th.csv",
    train_protocol="/dataset/Protocol-train.txt",
    val_protocol="/dataset/Protocol-val-labels.txt",
    root_dir="/data",
    val_bbox_csv="/dataset/val_bbox.csv",
)

# Device settings
device = "cuda"
model_input_size = (224, 224)

# Model settings
model = dict(
    model_name="convnextv2_tiny",
    pretrained=True,
    in_chans=3,
    num_classes=1,
)

# Training settings
train_multihead = False
cutmix_prob = 0.3
cutmix_alpha = 0.592

# Dataset settings
dataset = dict(
    oversample_live=False,
    val_max_per_subclass=300,
    val_seed=42,
    epoch_fraction=0.5,
)

# DataLoader settings
train_dataloader = dict(
    batch_size=32,
    shuffle=True,
    num_workers=16,
    drop_last=True,
    pin_memory=False,
    prefetch_factor=5,
)

val_dataloader = dict(
    batch_size=32,
    shuffle=False,
    num_workers=16,
    drop_last=False,
)

# Trainer settings
trainer = dict(
    type="Trainer",
    epochs=20,
    weights_save_folder="weights",
)

# Loss settings
loss = dict(
    type="binary_focal",
    alpha=0.528,
    gamma=0.73,
    contrastive=True,
    supcon_weight=0.306,
    supcon_temperature=0.141,
    supcon_proj_dim=128,
)

# Optimizer settings
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=1.1e-5,
)

# Scheduler settings
scheduler = dict(
    type="Cosine",
    cycle_mult=1.0,
    max_lr=1.8e-4,
    min_lr=6.8e-7,
    warmup_steps=0,
    gamma=1.0,
)

# Weights & Biases logging
wandb = dict(
    use=False,  # Set to True to enable W&B logging
    project_name="deepfake_detection",
    experiment_name="paired_contrastive",
)
