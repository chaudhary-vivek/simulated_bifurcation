import torch

def solve_ising(J, h, agents=10, steps=1000, dtype=torch.float32, device="cpu"):
    J = J.to(dtype=dtype, device=device)
    h = h.to(dtype=dtype, device=device)
    dim = J.shape[0]
    
    J_sym = (J + J.t()) / 2.0
    torch.diagonal(J_sym)[:] = 0.0
    
    pos = 2 * torch.rand(dim, agents, dtype=dtype, device=device) - 1
    mom = 2 * torch.rand(dim, agents, dtype=dtype, device=device) - 1
    
    dt = 0.05
    scale = 0.5 * (dim - 1)**0.5 / torch.sqrt((J_sym**2).sum())
    
    for step in range(steps):
        pressure = min(dt * step * 0.01, 1.0)
        mom += dt * (pressure - 1.0) * pos
        pos += dt * mom
        mom += dt * scale * (J_sym @ pos)
        pos.clamp_(-1.0, 1.0)
        mom[torch.abs(pos) > 1.0] = 0.0
    
    spins = torch.where(pos >= 0.0, 1.0, -1.0)
    energies = -0.5 * (spins.T @ J_sym @ spins).diag() + (h @ spins).T
    best_idx = energies.argmin()
    return spins[:, best_idx], energies[best_idx]



torch.manual_seed(42)
J = torch.tensor([[0, -2, 3], [-2, 0, 1], [3, 1, 0]])
h = torch.tensor([1, -4, 2])

spins, energy = solve_ising(J, h)

assert torch.equal(
    torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float32), spins
)
assert -11.0 == energy