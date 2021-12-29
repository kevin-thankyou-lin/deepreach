import torch

def air3D_dynamics(x, u, d, v_e=0.75, v_p=0.75):
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

def air3D_continuous_dynamics_from_discrete(f_discrete, x_t, u_t, d_t, dt=0.1):
    """Starting from discrete dynamics f_dis(x_t, u_t) = x_t+1, compute continuous dynamics f_cont(x_t, u_t) = \dot{x} using
    an euler approximation:

    f_cont(x_t, u_t) = \dot{x} \approx \frac{x_t+1 - x_t}{dt} = \frac{f_dis(x_t, u_t) - x_t}{dt}
    
    :params:
        f_dis: discrete dynamics function f_dis(x_t, u_t)
        x_t: state vector np.ndarray (meta_batch_size, N, 3)
        u_t: control vector np.ndarray (N_u, N_d)
        dt: time step
    
    :returns:
        x_dot: state vector np.ndarray (meta_batch_size, N, 3)
    """
    x_t_plus_1 = f_discrete(x_t, u_t, d_t)
    x_dot = (x_t_plus_1 - x_t[:, :, None, None, :]) / dt
    return x_dot

def air3D_discrete_dynamics(x, u, d, v_e=0.75, v_p=0.75, dt=0.1):
    """
    Starting from continous dynamics f_cont(x, u) = x_dot, compute discrete dynamics f_dis(x_t, u_t) = x_t+1 using an euler approximation:

    f_dis(x_t, u_t) = x_t+1 \approx x_t + dt * f_cont(x_t, u_t)

    :params:
        x: state vector np.ndarray (meta_batch_size, N, 3)
        u: control vector np.ndarray (N_u, N_d)
        d: disturbance vector np.ndarray (N_u, N_d)
        v_e: evader velocity vector
        v_p: pursuer velocity vector
        dt: time step
    
    :returns:
        x_t_plus_1: state vector np.ndarray (meta_batch_size, N, 3)
    """
    x_dot = air3D_dynamics(x, u, d, v_e, v_p)
    x_t_plus_1 = x[:, :, None, None, :] + dt * x_dot
    return x_t_plus_1

    