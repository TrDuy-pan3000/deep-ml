import torch

def smooth_labels(y_true: torch.Tensor, num_classes: int, epsilon: float) -> torch.Tensor:
    """
    Create smoothed one-hot target vectors.

    Args:
        y_true: torch.Tensor of shape (N,) with values in [0, K-1]
        num_classes: int, total number of classes (K)
        epsilon: float in [0, 1]

    Returns:
        torch.Tensor of shape (N, K) with smoothed probabilities.
    """
    n=len(y_true)
    logit = torch.zeros(n,num_classes)
    for i in range(n):
        logit[i,y_true[i]]=1-epsilon
    result = logit +epsilon/num_classes
    return result


def label_smoothing_cross_entropy(logits: torch.Tensor, y_true: torch.Tensor, num_classes: int,
                                   epsilon: float = 0.1, round_decimals: int = None) -> float:
    """
    Compute mean cross-entropy between logits and smoothed targets using stable log-softmax.

    Args:
        logits: torch.Tensor of shape (N, K), model output scores.
        y_true: torch.Tensor of shape (N,), integer class indices.
        num_classes: int, number of classes (K).
        epsilon: float in [0, 1].
        round_decimals: int | None, round the loss to this many decimals if given.

    Returns:
        float: Mean cross-entropy loss.
    """
    p = torch.softmax(logits, dim=1)
    l = -torch.sum(smooth_labels(y_true,num_classes,epsilon)*torch.log(p),dim=1)
    meanl= torch.mean(l)
    return meanl.tolist()