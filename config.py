# coding=utf8
import torch


class GlobalConfig:
    """
    Experimental Hyperparameters and System Configuration.
    Abstracted from the main logic to ensure modularity.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SIZE = (512, 512)
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # Structural parameters for BD-ResTransUNet
    EMBED_DIM = 768
    NUM_HEADS = 12
    DEFORMABLE_GROUPS = 4

    # Optimization (Abstracted for GitHub)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    @staticmethod
    def get_summary():
        return f"Config: {GlobalConfig.INPUT_SIZE}, Device: {GlobalConfig.DEVICE}"