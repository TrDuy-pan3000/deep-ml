import torch

def rnn_forward(input_sequence: list, initial_hidden_state: list, Wx: list, Wh: list, b: list) -> torch.Tensor:
    """
    Implements a simple RNN cell forward pass using PyTorch.

    Args:
        input_sequence: List of input vectors for each time step.
        initial_hidden_state: The initial hidden state vector.
        Wx: Weight matrix for input-to-hidden connections.
        Wh: Weight matrix for hidden-to-hidden connections.
        b: Bias vector.

    Returns:
        torch.Tensor: The final hidden state after processing the entire sequence,
                      rounded to four decimal places.
    """
    i = torch.tensor(input_sequence, dtype = float)
    h = torch.tensor(initial_hidden_state, dtype = float)
    wx  = torch.tensor(Wx, dtype = float)
    wh = torch.tensor(Wh, dtype = float)
    bias = torch.tensor(b, dtype = float)

    for x in i:
        h = torch.tanh(wx @ x + wh@h +bias)
    return h
    pass