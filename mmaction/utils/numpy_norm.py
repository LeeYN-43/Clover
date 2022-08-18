import numpy as np
import torch
import torch.nn.functional as F

def normalize_fn(x, axis=-1, order=2):
  l2 = np.atleast_1d(np.linalg.norm(x, ord=order, axis=axis))
  l2[l2 == 0] = 1
  return x / np.expand_dims(l2, axis=axis)

if __name__ == '__main__':
  a = np.random.rand(1000, 512)
  b = torch.Tensor(a)
  c = F.normalize(b, dim=-1)
  d = normalize_fn(a)
  ssss = 1