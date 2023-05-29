# MINIMIZE KL-DIVERGENCE & ELIMINATE epsilon PARAMETER
import torch
from torch import optim, nn


# KL-divergence for two covariance matrices
def kl_divergence(sigma_1, sigma_2):
    sigma_2_inv = torch.inverse(sigma_2)
    D = sigma_1.shape[0]
    kl = 0.5 * (torch.log(torch.det(sigma_2)) - torch.log(torch.det(sigma_1))
                + torch.sum(torch.multiply(sigma_2_inv, sigma_1)) - D)
    # instead of trace of product -> sum of element wise product
    return kl


# create observed covariance matrix for given L
def fill_cov(par1):
    D = len(par1)
    cov = torch.zeros(D, D)
    with torch.no_grad():
        par2 = torch.sqrt(1 - (par1 ** 2))  # epsilon = sgrt(1-L^2)

    for i in range(D):
        cov[i][i] = 1
        for j in range(D):
            if i != j:
                cov[i][j] = (par1[i] * par1[j]) / (
                            torch.sqrt(par1[i] ** 2 + par2[i] ** 2) * torch.sqrt(par1[j] ** 2 + par2[j] ** 2))
    return cov


# observed cov matrix for given L_data
L_0 = torch.randn(4)
L_data = torch.tanh(L_0)  # observed lambdas
Cov_data = fill_cov(L_data)  # observed cov matrix


# randomly generated cov matrix
D = 4
A = torch.randn(D, D)
B = torch.mm(A, torch.transpose(A, 0, 1))
Cov_data_random = torch.zeros(D, D)
for i in range(D):
    for j in range(D):
        Cov_data_random[i][j] = B[i][j] / (torch.sqrt(B[i][i]) * torch.sqrt(B[j][j]))


class Model(nn.Module):
    def __init__(self, size=4):
        super().__init__()
        self.D = size
        L_0 = torch.randn(size)
        self.L_un = nn.Parameter(L_0)

    @property
    def L(self):
        return torch.tanh(self.L_un)

    def forward(self):
        return fill_cov(self.L)


# Train model to minimize loss, that is, minimize kl-divergence
model = Model(size=D)
optimizer = optim.Adam(model.parameters(), lr=0.05)
# optimizer = optim.SGD(model.parameters(), lr=0.05)

niter = 100
loss = []
for _ in range(0, niter):
    optimizer.zero_grad()
    S_theta = model()
    loss_fn = kl_divergence(Cov_data, S_theta)
    loss_fn.backward()
    optimizer.step()
    loss.append(loss_fn.item())

