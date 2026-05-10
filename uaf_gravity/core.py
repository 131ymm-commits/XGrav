import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from typing import List

@dataclass
class TensorObject:
    """Основной класс тела"""
    id: int
    mass: float
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)


def nbody_derivatives(t, y, masses):
    n = len(masses)
    pos = y[:3*n].reshape(n, 3)
    vel = y[3*n:].reshape(n, 3)
    acc = np.zeros_like(pos)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = pos[j] - pos[i]
            r = np.linalg.norm(delta)
            if r < 1e-8:
                continue
            acc[i] += masses[j] * delta / (r ** 3)

    return np.concatenate([vel.ravel(), acc.ravel()])


def integrate_system(objects: List[TensorObject], t_span=(0, 6500), n_steps=1000):
    n = len(objects)
    masses = np.array([obj.mass for obj in objects])
    y0 = np.concatenate([obj.position.ravel() for obj in objects] +
                        [obj.velocity.ravel() for obj in objects])

    t_eval = np.linspace(t_span[0], t_span[1], n_steps)

    sol = solve_ivp(lambda t, y: nbody_derivatives(t, y, masses),
                    t_span, y0, t_eval=t_eval, method='DOP853',
                    rtol=1e-8, atol=1e-8)

    pos = sol.y[:3*n].reshape(n, 3, -1)
    vel = sol.y[3*n:].reshape(n, 3, -1)
    return sol.t, pos, vel


def detect_resonance_ratio(period1: float, period2: float, tolerance=0.085):
    """Определяет силу ближайшего резонанса"""
    if period1 < 1e-6 or period2 < 1e-6:
        return 0.0
    ratio = period1 / period2
    simple_ratios = [(1,1), (2,1), (3,1), (3,2), (4,1), (4,3), (5,2), (5,3), (7,3)]
    
    best_score = 0.0
    for p, q in simple_ratios:
        ideal = p / q
        closeness = abs(ratio - ideal) / ideal
        if closeness < tolerance:
            score = (tolerance - closeness) / tolerance
            best_score = max(best_score, score)
    return best_score
