# @title LiDAR
# LiDAR: Sensing Linear Probing Performance In Joint Embedding SSL Architectures https://arxiv.org/pdf/2312.04000
# https://github.com/rbalestr-lab/stable-ssl/blob/main/stable_ssl/monitors.py#L106

def LiDAR(sx, eps=1e-7, delta=1e-3):
    sx = sx.unflatten(0, (-1, 2))
    n, q, d = sx.shape
    mu_x = sx.mean(dim=1) # mu_x # [n,d]
    mu = mu_x.mean(dim=0) # mu # [d]

    diff_b = (mu_x - mu).unsqueeze(-1) # [n,d,1]
    S_b = (diff_b @ diff_b.transpose(-2,-1)).sum(0) / (n - 1) # [n,d,d] -> [d,d]
    diff_w = (sx - mu_x.unsqueeze(1)).reshape(-1,d,1) # [n,q,d] -> [nq,d,1]
    S_w = (diff_w @ diff_w.transpose(-2,-1)).sum(0) / (n * (q - 1)) + delta * torch.eye(d, device=sx.device) # [nq,d,d] -> [d,d]

    eigvals_w, eigvecs_w = torch.linalg.eigh(S_w)
    eigvals_w = torch.clamp(eigvals_w, min=eps)

    invsqrt_w = (eigvecs_w * (1. / torch.sqrt(eigvals_w))) @ eigvecs_w.transpose(-2,-1)
    S_lidar = invsqrt_w @ S_b @ invsqrt_w
    lam = torch.linalg.eigh(S_lidar)[0].clamp(min=0)
    p = lam / lam.sum() + eps
    return torch.exp(-torch.sum(p * torch.log(p)))

# sx = torch.randn(32, 128)
# print(sx.shape)
# lidar = LiDAR(sx)
# print(lidar.item())

