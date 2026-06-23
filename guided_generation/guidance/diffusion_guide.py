import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionGuide(nn.Module):

    def __init__(self):
        """
        A guide which has two parts
            1. A model which segments an image to logits.
            2. A loss function which provides the guidance term
        """
        super().__init__()
        self.segmenter = None
        self.loss = None


def shannon_entropy_loss(logits: torch.Tensor, mask: torch.Tensor = None):
    """Calc average entropy for the masked region."""
    pixel_entropy = calculate_shannon_entropy(logits=logits)
    if isinstance(mask, torch.Tensor):
        pixel_entropy = pixel_entropy[mask]
    # return negative entropy, as loss will always be minimized
    return -torch.mean(pixel_entropy)


def calculate_shannon_entropy(logits: torch.Tensor):
    """Calculate pixelwise shannon entropy."""
    squeeze = len(logits.shape) == 3
    if squeeze:
        logits = logits.unsqueeze(0)
    probabilities = F.softmax(logits, dim=1)  # Softmax across channels
    pixel_entropy = -probabilities * torch.log(probabilities + 1e-6)
    pixel_entropy = torch.sum(pixel_entropy, dim=1)
    if squeeze:
        pixel_entropy = pixel_entropy[0]
    return pixel_entropy
