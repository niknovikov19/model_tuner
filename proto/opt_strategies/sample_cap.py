import numpy as np
from scipy.stats import beta


def sample_spherical_cap(n, cos_phi, size=1):
    """
    Sample `size` points uniformly from the spherical cap on S^{n-1} defined by
    x[0] >= cos_phi (i.e. polar angle <= phi), centered on the +e1 axis.
    
    Parameters
    ----------
    n : int
        Ambient dimension.
    cos_phi : float
        Cosine of the cap half‐angle (0 <= cos_phi <= 1).
    size : int
        Number of samples to draw.
    
    Returns
    -------
    X : ndarray, shape (size, n)
        Sampled points on the unit sphere S^{n-1} within the cap.
    """
    # 1. Setup Beta parameters
    alpha = (n - 1) / 2.0
    beta_param = 0.5
    
    # 2. Compute normalization cutoff for truncated Beta CDF
    y_max = 1.0 - cos_phi**2
    M = beta.cdf(y_max, alpha, beta_param)
    
    # 3. Draw truncated Beta samples for y = 1 - z^2
    u = np.random.rand(size)
    y = beta.ppf(u * M, alpha, beta_param)
    
    # 4. Recover z = x[0] >= cos_phi
    z = np.sqrt(1 - y)
    
    # 5. Sample remaining coordinates on S^{n-2}
    #    by normalizing Gaussian draws
    G = np.random.randn(size, n - 1)
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    # Scale to radius sqrt(1 - z^2)
    coords = G * np.sqrt(1 - z**2)[:, None]
    
    # 6. Combine z and the rest
    X = np.concatenate([z[:, None], coords], axis=1)
    return X

# Example usage:
if __name__ == "__main__":
    n = 100      # dimension
    phi = np.pi / 6  # 30 degree cap
    cos_phi = np.cos(phi)
    samples = sample_spherical_cap(n, cos_phi, size=1000)
    print("Sample shape:", samples.shape)
    # Verify dot with center >= cos_phi
    print("Min dot:", np.min(samples.dot(np.eye(n)[0])))
