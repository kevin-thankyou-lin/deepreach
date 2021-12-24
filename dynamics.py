import torch

def air3D_dynamics(x, u, d, v_e, v_p):
    """x_dot = f(x, u, d)
    x: state vector np.ndarray (meta_batch_size, N, 3)
    u: control vector np.ndarray (N_u, N_d)
    d: disturbance vector np.ndarray (N_u, N_d)
    v_e: evader velocity vector
    v_p: pursuer velocity vector

    :returns: x_dot: np.ndarray (meta_batch_size, N, N_u, N_d, 3)
    """
    w_e, w_p = u, d
    # get correct shapes (not sure if it's necessary to do the repeats ...)
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    x1 = x1[..., None, None]
    # x1 = x1.repeat(1, 1, w_e.shape[0], w_e.shape[1])
    x2 = x2[..., None, None]
    # x2 = x2.repeat(1, 1, w_e.shape[0], w_e.shape[1])
    x3 = x3[..., None, None]
    # x3 = x3.repeat(1, 1, w_e.shape[0], w_e.shape[1])

    x1_dot = -v_e + v_p * torch.cos(x3) + w_e * x2
    x2_dot = v_p * torch.sin(x3) - w_e * x1
    x3_dot = (w_p - w_e)[None, None, :, :]
    
    x3_dot = x3_dot.repeat(1, x.shape[1], 1, 1)
    x_dot = torch.cat((x1_dot[..., None], x2_dot[..., None], x3_dot[..., None]), dim=-1)
    assert x_dot.shape[0] == x.shape[0] and x_dot.shape[1] == x.shape[1] and  x_dot.shape[-1] == x.shape[-1], "incorrect shapes"
    return x_dot