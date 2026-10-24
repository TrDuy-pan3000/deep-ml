import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	result = np.mean(np.mean(x,axis=1),axis=0)
	return result