import numpy as np

def integrate_abp_polar_leapfrog(
    r0,
    s0,
    phi0,
    n_steps,
    dt,
    gamma,
    D,
    seed=None,
    eps=1e-10,
):
    """
    Stochastic leapfrog / kick-drift-kick integrator for the polar Langevin system:
    """

    rng = np.random.default_rng(seed)

    time = np.zeros(n_steps)
    r = np.zeros((n_steps, 2))
    s = np.zeros(n_steps)
    phi = np.zeros(n_steps)

    r[0] = np.asarray(r0, dtype=float)
    s[0] = abs(s0)
    phi[0] = phi0

    # Noise amplitude for each half-step:
    # sqrt(2D * dt/2) = sqrt(D dt)
    sigma_half = np.sqrt(D * dt)

    def safe_speed(speed):
        return max(abs(speed), eps)

    def normalize_speed_angle(speed, angle):
        """
        Enforce positive speed. A negative speed with angle phi is equivalent to
        a positive speed with angle phi + pi.
        """
        if speed < 0:
            return -speed, angle + np.pi
        return speed, angle

    for i in range(n_steps - 1):

        time[i + 1] = time[i] + dt

        # -------------------------
        # First half-kick
        # -------------------------
        eta_s_1 = rng.normal()
        eta_phi_1 = rng.normal()

        s_i_safe = safe_speed(s[i])

        s_half = (
            s[i]
            + 0.5 * dt * (-gamma * s[i] + D / s_i_safe)
            + sigma_half * eta_s_1
        )

        phi_half = (
            phi[i]
            + sigma_half * eta_phi_1 / s_i_safe
        )

        s_half, phi_half = normalize_speed_angle(s_half, phi_half)

        # -------------------------
        # Drift
        # -------------------------
        direction_half = np.array([
            np.cos(phi_half),
            np.sin(phi_half),
        ])

        r[i + 1] = r[i] + s_half * direction_half * dt

        # -------------------------
        # Second half-kick
        # -------------------------
        eta_s_2 = rng.normal()
        eta_phi_2 = rng.normal()

        s_half_safe = safe_speed(s_half)

        s_new = (
            s_half
            + 0.5 * dt * (-gamma * s_half + D / s_half_safe)
            + sigma_half * eta_s_2
        )

        phi_new = (
            phi_half
            + sigma_half * eta_phi_2 / s_half_safe
        )

        s_new, phi_new = normalize_speed_angle(s_new, phi_new)

        s[i + 1] = s_new
        phi[i + 1] = phi_new

    return time, r, s, phi