import torch

def clip_grad_norm(parameters, max_norm: float) -> float:
    # TODO: compute total grad norm, scale in-place if it exceeds max_norm, return original norm
    gradient = []
    for p in parameters:
        if p.grad is not None:
            gradient.append(p.grad)
    total = 0
    for g in gradient:  
        total+=torch.sum(g**2)
    can = total**0.5
    if (can>max_norm):
        scale = max_norm/can
        with torch.no_grad():
            for g in gradient:
                g.mul_(scale)
    return can.tolist()
