"""Memory-light versions of unimatch's two local correlations, for the ONNX export only.

Both originals sample feature1 at every point's 81 window offsets in one grid_sample, which
materialises [B, C, H*W, 81] (455.6 MiB at the 1/4 scale of 576x320) and then a transposed copy of
it. Vulkan only guarantees 128 MiB for one storage buffer, so the exported graph could not run on
many GPUs. Here the 81 offsets are taken 9 at a time (one window row): the same grid_sample and
matmul on a slice of the same sampling grid, the 9 results concatenated. Every value is computed
the same way as before; nothing has parameters, so the weights are unchanged.
Same signatures and results as unimatch.matching (AdrianEddy/unimatch da140fa).
"""
import torch
import torch.nn.functional as F

from unimatch.geometry import coords_grid, generate_window_grid, normalize_coords


def _windowed_corr(feature0, feature1, sample_coords_norm, padding_mode, step):
    b, c, h, w = feature0.size()
    feature0_view = feature0.permute(0, 2, 3, 1).reshape(b, h * w, 1, c)  # [B, H*W, 1, C]
    parts = []
    for s in range(0, sample_coords_norm.size(2), step):
        window_feature = F.grid_sample(feature1, sample_coords_norm[:, :, s:s + step],
                                       padding_mode=padding_mode, align_corners=True
                                       ).permute(0, 2, 1, 3)  # [B, H*W, C, step]
        parts.append(torch.matmul(feature0_view, window_feature).view(b, h * w, -1))
    return torch.cat(parts, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]


def local_correlation_softmax(feature0, feature1, local_radius, padding_mode='zeros'):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]
    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1
    window_grid = generate_window_grid(-local_radius, local_radius, -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]
    sample_coords_softmax = sample_coords
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)
    valid = valid_x & valid_y
    sample_coords_norm = normalize_coords(sample_coords, h, w)
    corr = _windowed_corr(feature0, feature1, sample_coords_norm, padding_mode, local_w)
    corr[~valid] = -1e9
    prob = F.softmax(corr, -1)
    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(
        b, h, w, 2).permute(0, 3, 1, 2)
    flow = correspondence - coords_init
    match_prob = prob
    return flow, match_prob


def local_correlation_with_flow(feature0, feature1, flow, local_radius, padding_mode='zeros', dilation=1):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)
    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1
    window_grid = generate_window_grid(-local_radius, local_radius, -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)
    sample_coords = coords.unsqueeze(-2) + window_grid * dilation
    if not isinstance(flow, float):
        sample_coords = sample_coords + flow.view(b, 2, -1).permute(0, 2, 1).unsqueeze(-2)
    else:
        assert flow == 0.
    sample_coords_norm = normalize_coords(sample_coords, h, w)
    corr = _windowed_corr(feature0, feature1, sample_coords_norm, padding_mode, local_w)
    corr = corr.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # [B, (2R+1)^2, H, W]
    return corr


def install():
    """Swap the two functions where unimatch's model looks them up (imported by name)."""
    import unimatch.matching as matching
    import unimatch.unimatch as model
    for mod in (matching, model):
        assert hasattr(mod, "local_correlation_softmax") and hasattr(mod, "local_correlation_with_flow")
        mod.local_correlation_softmax = local_correlation_softmax
        mod.local_correlation_with_flow = local_correlation_with_flow
