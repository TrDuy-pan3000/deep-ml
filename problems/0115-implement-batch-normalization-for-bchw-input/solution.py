import numpy as np


def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray = None,
    running_var: np.ndarray = None,
    momentum: float = 0.1,
    epsilon: float = 1e-5,
    training: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stats_shape = (1, X.shape[1], 1, 1)

    if running_mean is None:
        running_mean = np.zeros(stats_shape, dtype=X.dtype)

    if running_var is None:
        running_var = np.ones(stats_shape, dtype=X.dtype)

    if training:
        # Tính thống kê riêng cho từng kênh.
        mean = X.mean(axis=(0, 2, 3), keepdims=True)
        var = X.var(axis=(0, 2, 3), keepdims=True, ddof=0)

        # Cập nhật thống kê để dùng khi inference.
        running_mean = (
            (1 - momentum) * running_mean + momentum * mean
        )
        running_var = (
            (1 - momentum) * running_var + momentum * var
        )
    else:
        # Khi inference, dùng thống kê đã lưu.
        mean = running_mean
        var = running_var

    normalized = (X - mean) / np.sqrt(var + epsilon)
    output = gamma * normalized + beta

    return output, running_mean, running_var