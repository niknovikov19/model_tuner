import numpy as np
import pandas as pd
import xarray as xr
from skimage.measure import find_contours


def _axis_from_da(da, dim):
    """Return 1D coordinate array for a given dimension (fallback to index)."""
    if dim in da.coords and da.coords[dim].ndim == 1:
        return da.coords[dim].values
    # fallback: evenly spaced indices
    return np.arange(da.sizes[dim], dtype=float)

def _pixel_to_data_coords(contour_rc, y_axis, x_axis):
    """
    Map 'row, col' (fractional pixel indices from skimage) to data coords.
    Works for non-uniform, monotonic axes using 1D interpolation.
    """
    r = contour_rc[:, 0]
    c = contour_rc[:, 1]
    # pixel indices go from 0..N-1; use fractional index to interpolate axis
    yi = np.interp(r, np.arange(len(y_axis), dtype=float), y_axis)
    xi = np.interp(c, np.arange(len(x_axis), dtype=float), x_axis)
    return np.column_stack([xi, yi])  # (x, y)

def _contours_at_level(da, level):
    """
    Get a list of polylines (each Nx2 in pixel coords (row, col)) for the given level.
    Ignores NaNs/inf by masking them out.
    """
    A = np.asarray(da.values, dtype=float)
    mask = ~np.isfinite(A)
    # skimage.find_contours expects high=8-connectivity when shapes are thin
    # Use mask to prevent contouring through invalid pixels
    return find_contours(A, level=level, fully_connected='high', positive_orientation='low', mask=~mask)

def _segment_intersections(p1, p2, eps=1e-9):
    """
    Compute intersection point of two segments p1=(x1,y1)->(x2,y2), p2=(x3,y3)->(x4,y4).
    Returns (x,y) or None. Uses a robust cross-product formulation.
    """
    (x1, y1), (x2, y2) = p1
    (x3, y3), (x4, y4) = p2

    # Bounding-box quick reject
    if (max(x1, x2) + eps < min(x3, x4) or max(x3, x4) + eps < min(x1, x2) or
        max(y1, y2) + eps < min(y3, y4) or max(y3, y4) + eps < min(y1, y2)):
        return None

    # Represent segments as p = a + t*(b-a), q = c + u*(d-c)
    ax, ay = x1, y1
    bx, by = x2, y2
    cx, cy = x3, y3
    dx, dy = x4, y4

    r_x, r_y = (bx - ax), (by - ay)
    s_x, s_y = (dx - cx), (dy - cy)

    # r x s (2D cross)
    rxs = r_x * s_y - r_y * s_x
    q_p_x, q_p_y = (cx - ax), (cy - ay)
    qpxr = q_p_x * r_y - q_p_y * r_x

    if abs(rxs) < eps:
        # Colinear or parallel
        if abs(qpxr) >= eps:
            return None  # parallel, non-intersecting
        # Colinear: project and check overlap; return no single point (skip)
        return None

    t = (q_p_x * s_y - q_p_y * s_x) / rxs
    u = (q_p_x * r_y - q_p_y * r_x) / rxs

    if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
        x = ax + t * r_x
        y = ay + t * r_y
        return (x, y)

    return None

def _polyline_segments(poly_xy):
    """Yield consecutive segments from a polyline (N x 2 array)."""
    for i in range(len(poly_xy) - 1):
        yield (tuple(poly_xy[i]), tuple(poly_xy[i+1]))

def _dedup_points(points, tol=1e-6):
    """Deduplicate points with a tolerance."""
    if not points:
        return []
    arr = np.asarray(points, dtype=float)        # shape (N, 2)
    key = np.round(arr / tol).astype(np.int64)   # bucketize by tol
    _, idx = np.unique(key, axis=0, return_index=True)
    return arr[np.sort(idx)].tolist()

def contour_intersections(X1: xr.DataArray, X2: xr.DataArray,
                          x10: float, x20: float,
                          out_column_names: tuple[str, str] | None = None
                          ) -> pd.DataFrame:
    """
    Find intersections of contours X1==x10 and X2==x20.
    Returns a pandas.DataFrame with columns ('x','y')
        (or given by out_column_names) in data coordinates.
    Assumes X1 and X2 share the same 2D grid (same dims and coords).
    """
    # Basic checks
    if X1.dims != X2.dims or any(X1.sizes[d] != X2.sizes[d] for d in X1.dims):
        raise ValueError("X1 and X2 must share the same dims and grid shape.")
    if X1.ndim != 2:
        raise ValueError("X1 and X2 must be 2-D arrays.")
    
    if out_column_names is None:
        out_column_names = ['x', 'y']
    else:
        out_column_names = list(out_column_names)

    dim_y, dim_x = X1.dims  # convention: first dim is rows (y), second is cols (x)
    y_axis = _axis_from_da(X1, dim_y)
    x_axis = _axis_from_da(X1, dim_x)

    # Extract pixel-space contours
    C1_rc = _contours_at_level(X1, x10)  # list of (N_i x 2) arrays: rows, cols
    C2_rc = _contours_at_level(X2, x20)

    # Map to data coords (x,y)
    C1_xy = [_pixel_to_data_coords(c, y_axis, x_axis) for c in C1_rc]
    C2_xy = [_pixel_to_data_coords(c, y_axis, x_axis) for c in C2_rc]

    # Segment–segment intersections across all polyline pairs
    hits = []
    for poly1 in C1_xy:
        for poly2 in C2_xy:
            # Iterate with quick bbox reject at poly level
            x1min, y1min = poly1.min(0)
            x1max, y1max = poly1.max(0)
            x2min, y2min = poly2.min(0)
            x2max, y2max = poly2.max(0)
            if (x1max < x2min or x2max < x1min or y1max < y2min or y2max < y1min):
                continue

            for s1 in _polyline_segments(poly1):
                for s2 in _polyline_segments(poly2):
                    p = _segment_intersections(s1, s2)
                    if p is not None:
                        hits.append(p)

    # Deduplicate and return
    hits = _dedup_points(hits, tol=1e-6)
    df = pd.DataFrame(hits, columns=out_column_names)
    # Optional: sort for determinism
    if not df.empty:
        df = df.sort_values(out_column_names).reset_index(drop=True)
    return df
