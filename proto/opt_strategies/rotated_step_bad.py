import math

from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import norm


def closest_c_1(a, b, s, d, *, atol=1e-8):
    """
    Closest point c to the line (0,s) such that
        |c - a|          = m   (m > 0),
        ⟨(b-a)/m , c-a⟩  >= d  (0 ≤ d ≤ 1).

    Works in any dimension.
    """

    m = np.linalg.norm(b - a)

    # ---------- basic vectors ----------------------------------------
    a, b, s = map(np.asarray, (a, b, s))
    if m <= 0:
        raise ValueError("m must be positive")
    if not (0 <= d <= 1 + atol):
        raise ValueError("d must lie in [0,1]")

    p_hat = (b - a) / m
    s_hat = s / np.linalg.norm(s)

    a_par = np.dot(a, s_hat)
    alpha = a - a_par * s_hat          # ⟂ line
    A     = np.linalg.norm(alpha)

    p_par = np.dot(p_hat, s_hat)
    beta  = p_hat - p_par * s_hat
    B     = 0.0 if A < atol else np.dot(beta, alpha) / A

    # ---------- helpers ----------------------------------------------
    def dot_val(x):                    # x = u_parallel
        r = np.sqrt(max(0.0, 1.0 - x * x))
        return x * p_par - r * B       # = ⟨p̂ , u⟩

    def feasible(x):
        return dot_val(x) + atol >= d

    def dist(x):
        return abs(A - m * np.sqrt(max(0.0, 1.0 - x * x)))

    # ---------- 1.  try unconstrained minimiser ----------------------
    if A >= m:                         # sphere outside the line
        x_star = 0.0                   # r = 1
        if feasible(x_star):
            u = -alpha / A
            return a + m * u
    else:                              # line intersects the sphere
        r_star = A / m
        for x_star in (+np.sqrt(1 - r_star**2), -np.sqrt(1 - r_star**2)):
            if feasible(x_star):
                u = x_star * s_hat - r_star * alpha / A
                return a + m * u
        # otherwise fall through to boundary solve

    # ---------- 2. boundary  ⟨p̂ , u⟩ = d  ---------------------------
    c2 = p_par * p_par + B * B
    c1 = -2.0 * d * p_par
    c0 = d * d - B * B

    sols = []
    if c2 < atol:                      # degenerates to linear
        if abs(c1) > atol:
            x = -c0 / c1
            if -1 <= x <= 1 and feasible(x):
                sols.append(x)
    else:
        disc = c1 * c1 - 4.0 * c2 * c0
        if disc < -atol:
            raise RuntimeError("No feasible solution (check d)")
        disc = max(0.0, disc)
        for root in (
            (-c1 + np.sqrt(disc)) / (2 * c2),
            (-c1 - np.sqrt(disc)) / (2 * c2),
        ):
            if -1 <= root <= 1 and feasible(root):
                sols.append(root)

    if not sols:
        raise RuntimeError("No feasible solution (check inputs)")

    # pick root with smallest distance to the line
    x_best = min(sols, key=dist)
    r_best = np.sqrt(1.0 - x_best * x_best)
    u_best = x_best * s_hat - r_best * alpha / A
    return a + m * u_best


def closest_c_2(a, b, s, d, *, atol=1e-10, grid=2000, refine=60):
    """
    Smallest-distance point c to the line (0,s) that satisfies
        ‖c - a‖              = ‖b - a‖       (same radius as point b),
        ⟨ (b-a)/‖b-a‖ , c-a ⟩  ≥ d            ( *no* absolute value ).

    Inputs
    -------
    a, b, s  : 1-D numpy arrays of equal length
    d        : 0 ≤ d ≤ 1   (because ⟨p̂,u⟩ is a cosine)
    atol     : numerical tolerance (default 1e-10)
    grid     : coarse grid for the first search over x
    refine   : number of golden-section steps for local refinement

    Returns
    -------
    c        : numpy array – the required point on the sphere
    """

    # ------------- basic geometry ------------------------------------
    a, b, s = map(np.asarray, (a, b, s))
    m = np.linalg.norm(b - a)                          # sphere radius
    if m <= 0:
        raise ValueError("‖b-a‖ must be positive")
    if not (0.0 <= d <= 1.0 + atol):
        raise ValueError("d must lie in [0,1]")

    p_hat = (b - a) / m
    s_hat = s / np.linalg.norm(s)

    # components relative to the line
    a_par = np.dot(a, s_hat)
    alpha = a - a_par * s_hat
    A = np.linalg.norm(alpha)

    p_par = np.dot(p_hat, s_hat)
    beta = p_hat - p_par * s_hat
    Q = np.linalg.norm(beta)                          # = ‖β‖

    # build orthonormal basis {e1, e2} in the plane ⟂ ŝ
    if A > atol:
        e1 = alpha / A
    else:                                             # a already on the line
        # pick *any* axis ⟂ ŝ
        tmp = np.eye(len(a))[np.argmin(np.abs(s_hat))]
        e1 = tmp - np.dot(tmp, s_hat) * s_hat
        e1 /= np.linalg.norm(e1)
        A = 0.0

    B = np.dot(beta, e1)                              # β·e1
    S2 = Q * Q - B * B                                # β component ⟂ e1
    S = math.sqrt(max(0.0, S2))
    if S > atol:
        e2 = (beta - B * e1) / S
    else:                                             # β ∥ e1
        # choose a random axis orthogonal to both ŝ and e1
        tmp = np.eye(len(a))[np.argmax(np.abs(e1))]
        e2 = tmp - np.dot(tmp, s_hat) * s_hat - np.dot(tmp, e1) * e1
        e2 /= np.linalg.norm(e2)

    # pre-compute angle between −e1 and β
    cos_theta = -B / Q if Q > atol else -1.0
    sin_theta =  S / Q if Q > atol else  0.0
    sin_theta_abs = abs(sin_theta)

    # ---------- distance for a given x = u·ŝ (= cos of angle to the line)
    def dist_for_x(x):
        if not (-1.0 <= x <= 1.0):
            return math.inf

        r = math.sqrt(max(0.0, 1.0 - x * x))
        if r < atol:                                  # u parallel to the line
            if x * p_par + atol < d:
                return math.inf
            cos_phi = 1.0                             # w = −e1
        else:
            gamma = (d - x * p_par) / (r * Q)         # need cosψ ≥ γ
            if gamma > 1.0 + atol:                    # impossible
                return math.inf

            if gamma <= cos_theta + atol:             # φ = 0 already works
                cos_phi = 1.0
            else:                                     # rotate just enough
                gamma = min(max(gamma, -1.0), 1.0)    # clamp numerically
                cos_phi = (
                    gamma * cos_theta
                    + math.sqrt(1.0 - gamma * gamma) * sin_theta_abs
                )

        # distance^2 = ‖α + m r w‖²  with  w·α = −A cosφ
        return math.sqrt(A * A + m * m * r * r - 2.0 * m * r * A * cos_phi)

    # ---------- 1. coarse search over x --------------------------------
    xs  = np.linspace(-1.0, 1.0, grid + 1)
    dists = np.array([dist_for_x(x) for x in xs])
    if np.all(np.isinf(dists)):
        raise RuntimeError("No feasible point – try a smaller d")

    idx = int(np.nanargmin(dists))
    x_lo = max(-1.0, xs[max(idx - 1, 0)])
    x_hi = min( 1.0, xs[min(idx + 1, grid)])

    # ---------- 2. local golden-section refinement ---------------------
    φ_g = (math.sqrt(5.0) - 1.0) / 2.0   # 1/φ
    c1 = x_hi - φ_g * (x_hi - x_lo)
    c2 = x_lo + φ_g * (x_hi - x_lo)
    f1, f2 = dist_for_x(c1), dist_for_x(c2)

    for _ in range(refine):
        if f1 < f2:
            x_hi, c2, f2 = c2, c1, f1
            c1 = x_hi - φ_g * (x_hi - x_lo)
            f1 = dist_for_x(c1)
        else:
            x_lo, c1, f1 = c1, c2, f2
            c2 = x_lo + φ_g * (x_hi - x_lo)
            f2 = dist_for_x(c2)

    x_opt = 0.5 * (x_lo + x_hi)
    r_opt = math.sqrt(max(0.0, 1.0 - x_opt * x_opt))

    # ---------- 3. choose the corresponding optimal rotation φ --------
    if r_opt < atol:
        w = -e1                                           # u parallel to the line
    else:
        gamma = (d - x_opt * p_par) / (r_opt * Q)
        if gamma <= cos_theta + atol:                     # no rotation needed
            cos_phi, sin_phi = 1.0, 0.0
        else:
            gamma = min(max(gamma, -1.0), 1.0)            # clamp
            sin_phi_tmp = math.sqrt(1.0 - gamma * gamma)
            if sin_theta < 0:
                sin_phi_tmp = -sin_phi_tmp
            cos_phi = gamma
            sin_phi = sin_phi_tmp
        w = -cos_phi * e1 + sin_phi * e2

    # ---------- 4. build u and the point c -----------------------------
    u = x_opt * s_hat + r_opt * w
    c = a + m * u
    return c


def closest_c_(a, b, s, d):
    """Returns c*, the point on {c: |c-a|=1, (c-a)·(b-a)>=d}
       that minimises dist(c, line through 0 in direction s)."""
    s  = s / norm(s)
    dv = (b - a) / norm(b - a)        # ensure unit
    beta = dv @ s

    a_par = (a @ s) * s
    a_perp = a - a_par
    r = norm(a_perp)

    # 1. unconstrained minimiser on the whole sphere
    if r >= 1:                 # the line misses the centre
        u0 = -a_perp / r
    else:                      # the line intersects the sphere
        u0 = np.sqrt(1 - r*r) * s - a_perp

    # 2. Is it inside the dome?
    if u0 @ dv >= d:
        return a + u0          # done!

    # 3. Otherwise project onto the boundary circle
    P = np.eye(len(a)) - np.outer(dv, dv)
    s_perp = P @ s
    v      = 2*a - 2*(a @ s)*s        # v⊥s
    v_perp = P @ v

    # an orthonormal basis of the relevant 2-plane
    e1 = s_perp / norm(s_perp)
    w  = v_perp - (v_perp @ e1)*e1
    e2 = w / norm(w)

    R  = np.sqrt(1 - d*d)
    A  = R * norm(v_perp)
    B  = d * beta
    C  = R * norm(s_perp)

    # minimise one-variable quadratic
    theta0 = math.atan2(B*C, A - B*C)
    theta1 = theta0 + math.pi

    def make_c(theta):
        u = d*dv + R*(math.cos(theta)*e1 + math.sin(theta)*e2)
        return a + u
    
    def _dist_sq(c, s):
        return np.dot(c, c) - (np.dot(c, s))**2
    
    c0, c1 = make_c(theta0), make_c(theta1)

    c = c0 if _dist_sq(c0, s) <= _dist_sq(c1, s) else c1
    return c

def closest_c(a, b, s, d):
    m = norm(b - a)                    # sphere radius
    a_hat, b_hat = a / m, b / m        # scale down
    c_hat        = closest_c_(a_hat, b_hat, s, d)
    return m * c_hat                     # scale back


if __name__ == '__main__':

    s = np.array([1, 2])
    a = np.array([3, 3])
    ab = np.array([-0.1, -0.5])
    b = a + (ab / norm(ab)) * 3
    d = 0.7

    c = closest_c(a, b, s, 1 - d)
    print(c)

    x_range = np.array([0, 3.5])
    k = s[1] / s[0]

    sz = 8

    plt.figure()

    plt.plot(x_range, k * x_range, 'k-')

    plt.plot([a[0], b[0]], [a[1], b[1]], 'r-')
    plt.plot([a[0], c[0]], [a[1], c[1]], 'b-')

    plt.plot(a[0], a[1], 'k.', markersize=sz)
    plt.plot(b[0], b[1], 'r.', markersize=sz)
    plt.plot(c[0], c[1], 'b.', markersize=sz)

    off = 0.2
    text_par = {'ha': 'center', 'va': 'center'}
    plt.text(a[0] + off, a[1] + off, 'a', **text_par)
    plt.text(b[0] - off, b[1] - off, 'b', **text_par)
    plt.text(c[0] - off, c[1] + off, 'c', **text_par)
    plt.text(s[0] * 3 - off, s[1] * 3 + off, 's', **text_par)

    plt.gca().set_aspect('equal', adjustable='box')

    ba = (b - a) / np.linalg.norm(b - a)
    ca = (c - a) / np.linalg.norm(c - a)
    d_ = 1 - np.abs(np.dot(ba, ca))
    plt.title(f'(b-a, c-a) = 1 - {d_:.2f}')

    plt.show()