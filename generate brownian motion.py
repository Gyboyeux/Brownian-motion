"""
Simulazione del moto Browniano standard (processo di Wiener).

Un processo di Wiener W(t) ha le seguenti proprietà:
  - W(0) = 0
  - Incrementi indipendenti: W(t+dt) - W(t) ~ N(0, dt)
  - Traiettorie continue (ma non differenziabili)

Qui simuliamo il processo discretizzando il tempo in N passi
e accumulando incrementi gaussiani indipendenti.
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_brownian_motion(n_steps: int, n_paths: int, T: float = 1.0, seed: int | None = None):
    """
    Simula n_paths traiettorie di moto Browniano standard su [0, T].

    Parameters
    ----------
    n_steps : numero di passi temporali (discretizzazione)
    n_paths : numero di traiettorie indipendenti da simulare
    T       : orizzonte temporale totale
    seed    : seed per riproducibilità

    Returns
    -------
    t : array dei tempi, shape (n_steps + 1,)
    W : array delle traiettorie, shape (n_paths, n_steps + 1)
    """
    rng = np.random.default_rng(seed)

    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    # Incrementi indipendenti N(0, dt) per ogni passo e ogni traiettoria
    increments = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))

    # W(0) = 0, poi somma cumulativa degli incrementi
    W = np.zeros((n_paths, n_steps + 1))
    W[:, 1:] = np.cumsum(increments, axis=1)

    return t, W


def plot_paths(t, W, title="Moto Browniano standard"):
    plt.figure(figsize=(9, 5))
    for path in W:
        plt.plot(t, path, linewidth=0.8, alpha=0.8)

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("W(t)")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("brownian_motion.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    N_STEPS = 1000
    N_PATHS = 10
    T = 1.0

    t, W = simulate_brownian_motion(n_steps=N_STEPS, n_paths=N_PATHS, T=T, seed=42)

    # Sanity check empirico: a t fissato, Var(W(t)) deve essere ~ t
    check_t_idx = -1  # ultimo istante, t = T
    empirical_var = np.var(W[:, check_t_idx])
    print(f"Varianza empirica a t={T}: {empirical_var:.4f} (teorica: {T})")

    plot_paths(t, W)
