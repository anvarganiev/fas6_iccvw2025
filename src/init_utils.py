import os
import random
import datetime

import torch
import numpy as np


def init_wandb(config):
    """Initialize Weights & Biases logging."""
    import wandb

    now = datetime.datetime.now()
    ts = now.strftime("%d.%m.%Y %H:%M")
    name = f"{config.wandb['experiment_name']} from {ts}"
    project = config.wandb["project_name"]
    wandb.init(project=project, config=config, name=name)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
