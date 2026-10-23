import torch

def batch_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Perform Batch Normalization on a 4D tensor in BCHW format."""
    xmean=X.mean(dim=(0,2,3), keepdims=True)
    xvar =X.var(dim=(0,2,3),keepdims=True, unbiased = False)
    result = (X-xmean)/torch.sqrt(xvar-epsilon)
    return result*gamma+beta