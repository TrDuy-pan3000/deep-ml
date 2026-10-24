import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores = np.array(scores)
    result = scores - max(scores)
    result = np.exp(result)/np.sum(np.exp(result))
    return np.array(np.round(result,4))