from dataclasses import dataclass
import dataclasses
from typing import Literal


@dataclass
class TrainConfig:
    # Training
    epochs: int = 100
    lr: float = 1e-3
    device: Literal["cuda", "cpu"] = "cuda"

    # Checkpoints
    load_from_checkpoint: bool = False
    checkpoint_path: str = "./checkpoints"

    save_best_model: bool = True
    save_last_model: bool = True

    # Monitor metric for best model
    monitor: Literal["val_loss", "val_acc"] = "val_acc"

    # Plots
    save_plots: bool = True
    plot_path: str = "./plots"

    # WandB
    wandb_monitor: bool = True
    project_name: str = "AlexNet"
    run_name: str = "alextnet-cifar-10"

    img_size: int = 224

    save_csv: bool = True
    csv_path: str = "log.csv"


@dataclass
class DatasetConfig:
    # Dataset
    img_size: int = 224
    batch_size: int = 64

    # DataLoader
    train_shuffle: bool = True
    test_shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = True

    # Transform
    normalize: bool = True


@dataclass
class InferenceConfig:
    img_size: int = 224
    load_from_checkpoint: bool = True
    checkpoint_path: str = "./checkpoints/best.pt"
    dataset_images_path: str = "./images"
    plot_file_name: str = "sample.png"
    
    

    