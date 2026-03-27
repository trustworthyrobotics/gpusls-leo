
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

import gpu_sls.gpu_sqp
from gpu_sls.external.linearization_sls.test_path import remainder_bound_path_based
from gpu_sls.external.linearization_sls.neural_wrapper import make_remainder_bound_builder
from gpu_sls.external.linearization_sls.neural.load import load_model

@dataclass
class MPCConfig:
    n: int
    nu: int
    N: int
    W: jnp.ndarray
    u_ref: jnp.ndarray
    dt: float

def pack_dynamics_as_single_input(dynamics, nx: int, nu: int, *, parameter, t_dim: int = 1, t_as_scalar: bool = True):
    dyn = partial(dynamics, parameter=parameter)  # binds keyword-only arg

    D = nx + nu + t_dim

    def f_flat(z: jnp.ndarray) -> jnp.ndarray:
        x = z[:nx]
        u = z[nx:nx+nu]
        t_slice = z[nx+nu:nx+nu+t_dim]
        t = t_slice[0] if (t_dim == 1 and t_as_scalar) else t_slice
        return dyn(x, u, t)

    return f_flat, D


class GenericMPC:
    def __init__(
        self,
        sls_config, sqp_config, admm_config,
        config, dynamics, constraints, obstacles,
        cost, Q_bar, R_bar,
        num_constraints: int,
        disturbance,
        X_in, U_in,
        neural_dynamics: bool = False,
        shift: int = 1,
        model_dir = "",
    ):
        self.sls_config = sls_config
        self.sqp_config = sqp_config
        self.admm_config = admm_config
        self.config = config
        self.shift = shift
        self.obstacles = obstacles
        num_obstacles = self.obstacles.shape[0]
        self.h_ct_ws = jnp.zeros((config.N + 1, num_constraints - num_obstacles))
        self.beta_ws = jnp.ones((config.N + 1, config.N + 1, num_constraints - num_obstacles)) * 1e-10
        self.mu_ws = jnp.zeros((config.N + 1, num_constraints))
        self.Phi_x_ws = jnp.zeros((config.N + 1, config.N + 1, config.n, config.n))
        self.Phi_u_ws = jnp.zeros((config.N, config.N + 1, config.nu, config.n))
        self.E_prev = jnp.zeros((config.N + 1, config.n, 2 * config.n))

        self.U0 = U_in
        self.X0 = X_in
        self.V0 = jnp.zeros((config.N + 1, config.n))
        self.w = jnp.zeros((config.N + 1, num_constraints))
        self.y = jnp.zeros((config.N + 1, num_constraints))
        self.rho = jnp.asarray(self.admm_config.initial_rho, dtype=self.w.dtype)

        self.dynamics = dynamics
        self.constraints = constraints
        self.cost = cost
        self.disturbance = disturbance

        self.Q_bar = Q_bar
        self.R_bar = R_bar

        D = config.n + config.nu
        V = D + 1
        f_flat, D_flat = pack_dynamics_as_single_input(
            dynamics,
            nx=config.n,
            nu=config.nu,
            parameter=config.dt,
            t_dim=1,
            t_as_scalar=True
        )
        if not neural_dynamics:
            remainder_func = partial(remainder_bound_path_based, f_flat, state_dim=config.n)
        else:
            # TODO: Remove this hardcoded
            model = load_model(model_dir)
            split_budget = (5, 5, 4, 1, 4, 1, 1)
            remainder_func = make_remainder_bound_builder(model, split_budget=split_budget)
        splts_cfg = (4, 4, 4, 4)

        work = partial(
            gpu_sls.gpu_sqp.sqp,
            self.sls_config, self.sqp_config, self.admm_config,
            cost, dynamics,
            None,
            constraints, disturbance,
            remainder_func, splts_cfg,
            self.Q_bar, self.R_bar
        )
        self._solve = jax.jit(work)

    def run(self, x0: jnp.ndarray, reference: jnp.ndarray, parameter: Any):
        X, U, V, w, y, rho, backoffs, Phi_x, Phi_u, betaN, muN, EN = self._solve(
            reference,
            parameter,
            self.config.W,
            x0, self.X0, self.U0, self.V0,
            self.w, self.y, self.rho,
            self.obstacles,
            self.h_ct_ws, self.beta_ws, self.mu_ws, self.Phi_x_ws, self.Phi_u_ws, self.E_prev
        )
        self.E_prev = EN
        s = self.shift

        invalid = (
            jnp.any(~jnp.isfinite(U)) |
            jnp.any(~jnp.isfinite(X)) |
            jnp.any(~jnp.isfinite(V)) |
            jnp.any(~jnp.isfinite(w)) |
            jnp.any(~jnp.isfinite(y)) |
            jnp.any(~jnp.isfinite(backoffs)) |
            jnp.any(~jnp.isfinite(betaN)) |
            jnp.any(~jnp.isfinite(muN)) |
            jnp.any(~jnp.isfinite(Phi_x)) |
            jnp.any(~jnp.isfinite(Phi_u))
        )

        def shift_and_pad(arr, pad_value=None):
            if pad_value is None:
                tail = jnp.tile(arr[-1:], (s,) + (1,) * (arr.ndim - 1))
            else:
                tail = jnp.broadcast_to(
                    pad_value,
                    (s,) + arr.shape[1:]
                )
            return jnp.concatenate([arr[s:], tail], axis=0)

        # ---- primal warm starts ----
        self.U0 = jax.lax.cond(
            invalid,
            lambda _: jnp.tile(self.config.u_ref, (self.config.N, 1)),
            lambda _: shift_and_pad(U),
            operand=None,
        )

        self.X0 = jax.lax.cond(
            invalid,
            lambda _: jnp.tile(x0, (self.config.N + 1, 1)),
            lambda _: shift_and_pad(X),
            operand=None,
        )

        self.V0 = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros((self.config.N + 1, self.config.n), dtype=V.dtype),
            lambda _: shift_and_pad(V),
            operand=None,
        )

        # ---- constraint / tube warm starts ----
        self.h_ct_ws = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros_like(self.h_ct_ws),
            lambda _: shift_and_pad(backoffs),
            operand=None,
        )

        self.beta_ws = jax.lax.cond(
            invalid,
            lambda _: jnp.ones_like(self.beta_ws) * 1e-10,
            lambda _: shift_and_pad(betaN),
            operand=None,
        )

        self.mu_ws = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros_like(self.mu_ws),
            lambda _: shift_and_pad(muN),
            operand=None,
        )

        # ---- ADMM-ish dual warm starts ----
        self.w = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros_like(self.w),
            lambda _: shift_and_pad(w),
            operand=None,
        )

        self.y = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros_like(self.y),
            lambda _: shift_and_pad(y),
            operand=None,
        )

        rho = jnp.asarray(rho, dtype=self.rho.dtype)

        # Only rescale y if the solve was valid
        self.y = jax.lax.cond(
            invalid,
            lambda _: self.y,
            lambda _: rho / self.rho * self.y,
            operand=None,
        )

        self.rho = jax.lax.cond(
            invalid,
            lambda _: jnp.asarray(self.admm_config.initial_rho, dtype=self.rho.dtype),
            lambda _: rho,
            operand=None,
        )

        self.Phi_x_ws = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros_like(self.Phi_x_ws),
            lambda _: Phi_x,
            operand=None,
        )

        self.Phi_u_ws = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros_like(self.Phi_u_ws),
            lambda _: Phi_u,
            operand=None,
        )

        return U[0], X, U, V, backoffs, Phi_x, Phi_u, self.E_prev