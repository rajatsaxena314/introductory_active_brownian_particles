import numpy as np

def integrate_langevin_leapfrog(
    x0,
    v0,
    n_steps,
    dt,
    g,
    m,
    D,
    seed=None
):
    """
    Integrate a simple Langevin equation using a kick-drift-kick scheme.
    """

    if seed is not None:
        np.random.seed(seed)

    time = np.zeros(n_steps)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)

    x[0] = x0
    v[0] = v0

    for i in range(1, n_steps):

        time[i] = i * dt

        # Kick: half-step velocity update
        R1 = np.random.normal(0, 1)
        v_half = (
            v[i - 1]
            + 0.5 * dt * (-g * v[i - 1] / m)
            + 0.5 * np.sqrt(D * dt) * R1
        )

        # Drift: full-step position update
        x[i] = x[i - 1] + dt * v_half

        # Kick: second half-step velocity update
        R2 = np.random.normal(0, 1)
        v[i] = (
            v_half
            + 0.5 * dt * (-g * v_half / m)
            + 0.5 * np.sqrt(D * dt) * R2
        )

    return time, x, v