import matplotlib.pyplot as plt
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

Lex = ['flowers', 'grew', 'on', 'tall', 'towers', 'two']

plt.scatter(U[:, 0], U[:, 1], c='Blue')
plt.xlim([-0.6, -0.1])

for a in range(U.size()[0]):
    plt.text(0.01 + U[a, 0], U[a, 1], Lex[a])

plt.savefig('vectors.png')
plt.show()