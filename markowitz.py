import torch

def optimize_portfolio(cov, ret, risk=1, bits=3, agents=10):
    assets= cov.shape[0]
    q =  -0.5 * risk * cov
    l =  ret

    pos = 2 * torch.rand(assets, agents) - 1
    mom =  2 * torch.rand(assets, agents) - 1
    
    scale = 0.5 * (assets - 1)**0.5 / torch.sqrt((q**2).sum())
    dt = 0.5 * (assets - 1)**0.5 / torch.sqrt((q**2).sum())
    
    for step in range(1000):
        p = min(dt * step * 0.01, 1.0)
        mom += dt * (p - 1) * pos
        pos += dt * mom
        mom += dt * scale * q @ pos
        pos = torch.clamp(pos, -1, 1)
        mom[torch.abs(pos) > 1] = 0
    
    w = ((torch.where(pos >= 0, 1.0, -1.0) + 1) / 2) * (2**bits - 1)
    gains = torch.einsum('ij,j->i', w.T, ret) - 0.5 * risk * torch.einsum('ij,jk,ik->i', w.T, cov, w.T)
    return w[:, gains.argmax()], gains.max().item()



cov = torch.tensor([[1.0, 1.2, 0.7], [1.2, 1.0, -1.9], [0.7, -1.9, 1.0]])
ret = torch.tensor([0.2, 0.05, 0.17])
portfolio, gain = optimize_portfolio(cov, ret)
print(f"Optimized portfolio: {portfolio}\nExpected gain: {gain:.4f}")

assert torch.equal(torch.tensor([0.0, 7.0, 7.0]), portfolio)
