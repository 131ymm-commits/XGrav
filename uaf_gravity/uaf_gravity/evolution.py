import numpy as np
import random
from typing import List
from .core import TensorObject, integrate_system, detect_resonance_ratio


def compute_uaf_stability(objects: List[TensorObject], t_span=(0, 6500), n_steps=900):
    t, pos, vel = integrate_system(objects, t_span, n_steps)
    n_bodies = len(objects)
    masses = np.array([obj.mass for obj in objects])

    # UAF Gamma
    kinetic = np.zeros(len(t))
    potential = np.zeros(len(t))

    for step in range(len(t)):
        for i in range(n_bodies):
            v = vel[i, :, step]
            kinetic[step] += 0.5 * masses[i] * np.dot(v, v)
        
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                r = np.linalg.norm(pos[i, :, step] - pos[j, :, step])
                if r > 1e-6:
                    potential[step] += -masses[i] * masses[j] / r

    Ek = kinetic / (kinetic.max() + 1e-9)
    Ep = potential / (np.abs(potential).max() + 1e-9)
    gamma = np.exp(-np.mean(np.abs(Ek - Ep)))

    # Resonance Analysis
    central_mass = masses[0]
    resonance_bonus = 0.0
    danger_penalty = 0.0

    for i in range(1, n_bodies):
        r = np.linalg.norm(pos[i, :, -1])
        if r < 0.5:
            continue
        period = 2 * np.pi * np.sqrt(r**3 / central_mass)
        res_strength = detect_resonance_ratio(period, 1.0)
        
        if res_strength > 0.5:
            resonance_bonus += res_strength * 0.18
        if abs(period - 1/3) < 0.12 or abs(period - 1/2.5) < 0.12:
            danger_penalty += 0.35

    resonance_score = resonance_bonus - danger_penalty

    # Min distance
    min_dist = 1e9
    for step in range(len(t)):
        for i in range(1, n_bodies):
            for j in range(i + 1, n_bodies):
                d = np.linalg.norm(pos[i, :, step] - pos[j, :, step])
                min_dist = min(min_dist, d)

    stability = gamma * (1.0 + 0.6 * resonance_score) * np.exp(-max(0, 1.2 - min_dist))
    return stability, gamma, resonance_score, min_dist


class UAFGravityEvolution:
    def __init__(self, n_asteroids=124, pop_size=10, generations=20):
        self.n_asteroids = n_asteroids
        self.pop_size = pop_size
        self.generations = generations

    def create_asteroid_cloud(self, central_mass=1800.0):
        objects = [TensorObject(0, central_mass, np.zeros(3), np.zeros(3))]

        # 5 Shepherd Moons (крупные тела)
        for i in range(1, 6):
            r = 8 + i * 3.5
            pos = np.array([r, 0.0, np.random.normal(0, 0.8)])
            v_circ = np.sqrt(central_mass / r) * 1.02
            vel = np.array([0, v_circ, 0])
            objects.append(TensorObject(i, 25.0, pos, vel))

        # Мелкие астероиды
        for i in range(6, self.n_asteroids + 6):
            r = np.random.uniform(5.0, 26.0)
            theta = np.random.uniform(0, 2 * np.pi)
            z = np.random.normal(0, 1.6)
            pos = np.array([r * np.cos(theta), r * np.sin(theta), z])
            
            v_circ = np.sqrt(central_mass / r) * np.random.uniform(0.88, 1.15)
            vel_dir = np.cross(pos, [0, 0, 1])
            vel_dir /= np.linalg.norm(vel_dir) + 1e-9
            vel = vel_dir * v_circ + np.random.normal(0, 0.12, 3)

            objects.append(TensorObject(i, np.random.uniform(0.6, 2.8), pos, vel))

        return objects

    def evolve(self):
        population = [self.create_asteroid_cloud() for _ in range(self.pop_size)]
        best_config = None
        best_score = -np.inf
        history = []

        for gen in range(self.generations):
            scores = []
            for config in population:
                score, _, _, _ = compute_uaf_stability(config)
                scores.append(score)

            best_idx = np.argmax(scores)
            gen_best = scores[best_idx]

            if gen_best > best_score:
                best_score = gen_best
                best_config = population[best_idx]

            history.append(gen_best)
            print(f"Gen {gen+1:2d}/{self.generations} → Best: {gen_best:.4f} | Global: {best_score:.4f}")

            # Эволюция с элитизмом
            new_pop = [population[best_idx]]
            while len(new_pop) < self.pop_size:
                parent = random.choice(population)
                child = []
                for obj in parent:
                    noise_pos = np.random.normal(0, 0.15 if obj.mass > 10 else 0.22, 3)
                    noise_vel = np.random.normal(0, 0.08 if obj.mass > 10 else 0.13, 3)
                    new_obj = TensorObject(
                        obj.id,
                        max(0.4, obj.mass * np.random.uniform(0.93, 1.07)),
                        obj.position + noise_pos,
                        obj.velocity + noise_vel
                    )
                    child.append(new_obj)
                new_pop.append(child)
            population = new_pop

        return best_config, best_score, history
