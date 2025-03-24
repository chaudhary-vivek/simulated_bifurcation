import torch

def optimize_qubo(Q, minimize=False, agents=128, dtype=torch.float32, device="cpu"):
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor(Q)
    
    Q = Q.to(device=device, dtype=dtype)
    dim = Q.shape[0]
    
    pos = 2 * torch.rand(dim, agents, dtype=dtype, device=device) - 1
    mom = 2 * torch.rand(dim, agents, dtype=dtype, device=device) - 1
    
    max_steps = 5000
    dt = 0.01
    scale = 0.5 * (dim - 1) ** 0.5 / torch.sqrt((Q**2).sum())
    
    if minimize:
        Q = -Q
    
    Q_sym = 0.5 * (Q + Q.t())
    
    for step in range(max_steps):
        p = min(dt * step * 0.01, 1.0)
        mom += dt * (p - 1.0) * pos
        pos += dt * mom
        spins = torch.sign(pos)
        mom += dt * scale * (Q_sym @ spins)
        pos = torch.clamp(pos, -1.0, 1.0)
        mom[torch.abs(pos) > 1.0] = 0.0
    
    binary_values = (torch.sign(pos) + 1) / 2
    values = (binary_values.T @ Q @ binary_values).diag()
    best_idx = values.argmax()
    best_value = values[best_idx]
    
    if minimize:
        best_value = -best_value
    
    return binary_values[:, best_idx], best_value



P = 15.5

Q = torch.tensor([
    [2, -P, -P, 0, 0, 0],
    [0, 2, -P, -P, 0, 0],
    [0, 0, 2, -2 * P, 0, 0],
    [0, 0, 0, 2, -P, 0],
    [0, 0, 0, 0, 4.5, -P],
    [0, 0, 0, 0, 0, 3],
])

binary_vector, objective_value = optimize_qubo(Q, minimize=False, agents=10)

assert torch.equal(
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32), binary_vector
    )
assert 7.0 == objective_value