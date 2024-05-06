import torch as th
import torch.nn.functional as F

def sample_ray_points(ray_o, ray_d, near, far, N_samples=64):
    t_vals = th.linspace(0., 1., steps=N_samples).to(ray_o.device)
    z_vals = near.unsqueeze(-1) * (1.-t_vals) + far.unsqueeze(-1) * (t_vals) # [bs, N_rays, N_samples]

    # get intervals between samples
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = th.cat([mids, z_vals[...,-1:]], -1)
    lower = th.cat([z_vals[...,:1], mids], -1)
    # stratified samples in those intervals
    t_rand = th.rand(z_vals.shape).to(ray_o.device) 
    z_vals = lower + (upper - lower) * t_rand # [bs, N_rays, N_samples]
    pts = ray_o[...,None,:] + ray_d[...,None,:] * z_vals[...,:,None] # [bs, N_rays, N_samples, 3]
    
    viewdirs = ray_d / th.norm(ray_d, dim=-1, keepdim=True) # (bs, N_rays, 3)
    # viewdirs = th.reshape(viewdirs, [ray_d.shape[0],-1,3]) # (bs, N_rays, 3)

    return pts, viewdirs, z_vals

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    """
    # if not global_args.occupancy:
    raw2alpha = lambda raw, dists, act_fn=F.softplus: 1.-th.exp(-act_fn(raw)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = th.cat([dists, th.Tensor([1e10]).to(dists.device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * th.norm(rays_d[...,None,:], dim=-1)

    rgb = th.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = th.randn(raw[...,3].shape) * raw_noise_std
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * th.cumprod(th.cat([th.ones((alpha.shape[0], alpha.shape[1], 1)).to(dists.device), 1.-alpha + 1e-10], -1), -1)[:, :, :-1]

    # T_s = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], 1)).to(dists.device), 1.-alpha + 1e-10], -1), -1)[:, :, :-1]
    rgb_map = th.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    # depth_map = torch.sum(weights * z_vals, -1)
    # disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = th.sum(weights, -1)

    if white_bkgd:
        print("Using White Bkgd!!")
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, acc_map, alpha
