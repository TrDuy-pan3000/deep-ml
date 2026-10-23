import torch
import torch.nn as nn


def train_neuron(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial_weights: torch.Tensor,
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> tuple[list[float], float, list[float]]:
    X = features.float()
    y = labels.float()

    # Tạo tham số cần học, không thay đổi trọng số đầu vào.
    w = initial_weights.detach().clone().float().requires_grad_(True)
    b = X.new_tensor(initial_bias, requires_grad=True)

    criterion = nn.MSELoss()
    mse_values = []

    for _ in range(epochs):
        # Xóa gradient của epoch trước.
        w.grad = None
        b.grad = None

        # Dự đoán và tính sai số.
        predictions = torch.sigmoid(X @ w + b)
        loss = criterion(predictions, y)
        mse_values.append(round(loss.item(), 4))

        # Tính gradient rồi cập nhật tham số.
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

    updated_weights = [round(value, 4) for value in w.detach().tolist()]
    updated_bias = round(b.item(), 4)

    return updated_weights, updated_bias, mse_values