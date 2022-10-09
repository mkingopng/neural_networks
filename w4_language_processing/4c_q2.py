import torch

M = torch.Tensor(
    [[0, 1, 0, 1, 0, 1],
     [1, 0, 1, 1, 0, 1],
     [0, 1, 0, 2, 0, 1],
     [1, 1, 2, 0, 1, 2],
     [0, 0, 0, 1, 0, 1],
     [1, 1, 1, 2, 1, 0]])

U, S, V = torch.svd(M)

torch.set_printoptions(precision=2)

print(U)
print(S)
print(V)
