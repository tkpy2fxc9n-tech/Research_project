# Physical <-> nondimensional conversion for ONE beam scenario -- used only
# by the p14 runs (configs/runs/p8_nondim.yaml,
# dataset/make_dataset_nondim.py, checks/check_nondim_scaling.py). Every
# other run keeps using physical (E, rho, L) Config fields directly and
# never touches this module.
#
# x_nd = x/L, t_nd = t*c/L, u_nd = u/U0, with c = sqrt(E/rho). The PDE
# u_tt = (E/rho)*u_xx becomes u_nd,t_nd,t_nd = u_nd,x_nd,x_nd (coefficient
# 1) under this change of variables -- see the p14 planning discussion.
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np


@dataclass
class Scaling:
    L: float     # beam length (m)
    E: float     # Young's modulus (Pa)
    rho: float   # density (kg/m^3)
    U0: float    # reference displacement amplitude (m) -- the "1" of u_nd

    @property
    def c(self) -> float:
        return float(np.sqrt(self.E / self.rho))

    # --- physical -> nondimensional -------------------------------------
    def length_to_nd(self, x):
        return np.asarray(x) / self.L

    def time_to_nd(self, t):
        return np.asarray(t) * self.c / self.L

    def omega_to_nd(self, omega):
        # angular frequency: 1/time, so it rescales the opposite way of a
        # duration (sigma, t_end, ...).
        return np.asarray(omega) * self.L / self.c

    def amp_to_nd(self, u):
        return np.asarray(u) / self.U0

    # --- nondimensional -> physical -------------------------------------
    def length_from_nd(self, x_nd):
        return np.asarray(x_nd) * self.L

    def time_from_nd(self, t_nd):
        return np.asarray(t_nd) * self.L / self.c

    def omega_from_nd(self, omega_nd):
        return np.asarray(omega_nd) * self.c / self.L

    def amp_from_nd(self, u_nd):
        return np.asarray(u_nd) * self.U0


def physical_config(nd_cfg, scaling: Scaling):
    # Given a canonical nondimensional Config (E=rho=L=1) and a Scaling,
    # returns the physical Config representing the SAME discretized problem
    # for this beam's real (L, E, rho, U0) -- Nx/Nt/SS/M_BACK/N_FWD/ndt
    # (pure step/grid counts) are untouched; only quantities with physical
    # units (durations, frequencies, amplitudes) are rescaled.
    return dataclasses.replace(
        nd_cfg,
        L=scaling.L, E=scaling.E, rho=scaling.rho,
        t_end=float(scaling.time_from_nd(nd_cfg.t_end)),
        SIGMA_MIN=float(scaling.time_from_nd(nd_cfg.SIGMA_MIN)),
        SIGMA_MAX=float(scaling.time_from_nd(nd_cfg.SIGMA_MAX)),
        OMEGA_MIN=float(scaling.omega_from_nd(nd_cfg.OMEGA_MIN)),
        OMEGA_MAX=float(scaling.omega_from_nd(nd_cfg.OMEGA_MAX)),
        AMP_MIN=float(scaling.amp_from_nd(nd_cfg.AMP_MIN)),
        AMP_MAX=float(scaling.amp_from_nd(nd_cfg.AMP_MAX)),
    )
