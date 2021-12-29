import torch

from dynamics import air3D_dynamics

def air3D_hamiltonian_discrete(dudx, x, discrete_action_step, omega_max, v_e, v_p):
    """
    Calculates the Hamiltonian of the air3D system.
    
    H(x, t) = <∇_x V(x, t), x_dot>
    params:
        dudx: np.ndarray (N, 3) for ∇_x V(x, t)
        x1, x2, x3: state vectors np.ndarray (meta_batch_size, N, 3)
        omega_max: maximum control and disturbance inputs
        d: disturbance vector np.ndarray (N, 1)
        v_e: evader velocity vector np.ndarray (N, 1)
        v_p: pursuer velocity vector np.ndarray (N, 1)
        u: evader angular velocity vector np.ndarray (N, 1)
        d: pursuer angular velocity vector np.ndarray (N, 1)
    """
    # Initialize the Hamiltonian
    u = torch.arange(-omega_max, omega_max + discrete_action_step, discrete_action_step, device=x.device)
    d = torch.arange(-omega_max, omega_max + discrete_action_step, discrete_action_step, device=x.device)
    
    grid_u, grid_d = torch.meshgrid(u, d, device=x.device)

    # Calculate the Hamiltonian
    f_xud = air3D_dynamics(x, grid_u, grid_d, v_e, v_p) # f_xud should be (meta_batch_size, N, N_u, N_d, 3)
    assert f_xud.ndim == 5, "f_xud should be (meta_batch_size, N, N_u, N_d, 3)"
    dudx = dudx[:, :, None, None, :]
    f_dot_dudx = torch.sum(f_xud * dudx, dim=-1) # broadcasting along the second to last and third to last dimensions
    assert f_dot_dudx.shape[0] == x.shape[0] and f_dot_dudx.shape[1] == x.shape[1] and f_dot_dudx.shape[2] == grid_u.shape[0] and f_dot_dudx.shape[3] == grid_d.shape[1], "incorrect shapes"
    # need to perform the max and min operators over the correct dimensions of f_xud
    # H = torch.max(torch.min(f_dot_dudx, dim=-1).values, dim=-1).values (previously)
    H = torch.min(torch.max(f_dot_dudx, dim=-2).values, dim=-1).values
    assert H.shape[0] == x.shape[0] and H.shape[1] == x.shape[1], "incorrect shapes"
    return H
