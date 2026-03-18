import jax.numpy as jnp

def get_trajectory_tubes(Phi_x, E_prev):
    # Phi_x: (K, T, 2, nx)def get_tube_width(Phi_x, Phi_u, E):
    # Phi_x: [T+1, T+1, nx, nw]
    # Phi_u: [T,   T+1, nu, nw]
    # E:     [T+1, nw, ne]

    Phi_x_E = jnp.einsum("kjxn,jne->kjxe", Phi_x, E_prev)   # [T+1, T+1, nx, ne]
    return jnp.linalg.norm(Phi_x_E, ord=1, axis=-1).sum(axis=1)