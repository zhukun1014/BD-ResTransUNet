# coding=utf8
import torch


class ModelConstants:
    """
    Encapsulates architectural hyperparameters for BD-ResTransUNet.
    Abstracted to ensure logical isolation of structural constraints.
    """
    BACKBONE_CHANNELS = [64, 128, 256, 512]
    ATTENTION_HEADS = 8
    DEFORM_GROUPS = 4
    EXPANSION_RATIO = 4
    DROPOUT_RATE = 0.1
    ACTIVATION = "GELU"

    # Boundary Stream Config
    BOUNDARY_KERNEL_SIZE = 3
    BOUNDARY_CHANNELS = 32

    # Global System Settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRECISION = torch.float32