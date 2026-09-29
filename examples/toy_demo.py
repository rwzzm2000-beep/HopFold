"""Run the small public HopFold smoke example."""
import numpy as np
from hopfold import build_carrier, decode_carrier, hopfold_linear

x = np.array([[1., 0.], [0., 2.], [2., 1.], [1., 3.]], dtype=np.float32)
edges = np.array([[1, 0], [2, 0], [0, 1], [3, 1], [0, 2], [1, 3]], dtype=np.int64)
carrier = build_carrier(x, edges)
self_x, neighbor_x = decode_carrier(carrier)
y = hopfold_linear(carrier, np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32))
print({"codes_shape": carrier.codes.shape, "codebook_shape": carrier.codebook.shape})
print("decoded self:\n", np.round(self_x, 3))
print("decoded neighbor statistic:\n", np.round(neighbor_x, 3))
print("first-layer projection:\n", np.round(y, 3))
