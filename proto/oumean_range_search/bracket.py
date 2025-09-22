import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

LblL, LblM, LblR = 0, 1, 2  # tri-state

@dataclass
class JointIntervalFinder:
    # thresholds & conditions
    y1: float
    y2: float
    r1: float
    r2: float
    x2_mode: str = "min_ge"   # {"min_ge","min_le","y1_ge","y1_le","y2_ge","y2_le"}

    # batch & stopping
    N: int = 10
    tol_frac: float = 0.1
    max_iters: int = 100
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # start either with center+halfwidth or with endpoints
    x_start: Optional[float] = None
    w_start: Optional[float] = None
    x1_start: Optional[float] = None   # left endpoint (alias)
    x2_start: Optional[float] = None   # right endpoint (alias)

    # internal state
    _A: float = field(init=False, default=0.0)
    _B: float = field(init=False, default=1.0)
    _iter: int = field(init=False, default=0)
    _allL_run: int = field(init=False, default=0)
    _allR_run: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)

    _X: List[float] = field(default_factory=list, init=False)
    _F1: List[float] = field(default_factory=list, init=False)
    _F2: List[float] = field(default_factory=list, init=False)
    _L:  List[int]   = field(default_factory=list, init=False)

    # ---------- public API ----------
    def suggest_x_values(self) -> np.ndarray:
        if not self._initialized:
            # derive initial [A,B]
            if self.x1_start is not None and self.x2_start is not None:
                self._A, self._B = float(self.x1_start), float(self.x2_start)
                if self._A >= self._B:
                    raise ValueError("x1_start must be < x2_start")
            elif self.x_start is not None and self.w_start is not None:
                self._A = float(self.x_start) - float(self.w_start)
                self._B = float(self.x_start) + float(self.w_start)
            else:
                raise ValueError("Provide either (x_start,w_start) or (x1_start,x2_start).")
            self._initialized = True
            return np.linspace(self._A, self._B, num=self.N)

        X, F1, F2, L = self._current_arrays()
        x1_br, x2_br, _ = self._brackets(X, L)
        xs = self._choose_batch(self._A, self._B, self.N, x1_br, x2_br)
        return xs

    def process_probe_result(self, x_vals: np.ndarray, f1_vals: np.ndarray, f2_vals: np.ndarray) -> None:
        x_vals = np.asarray(x_vals); f1_vals = np.asarray(f1_vals); f2_vals = np.asarray(f2_vals)
        for x, f1, f2 in zip(x_vals, f1_vals, f2_vals):
            self._X.append(float(x)); self._F1.append(float(f1)); self._F2.append(float(f2))
            self._L.append(self._label(f1, f2))
        self._iter += 1
        X, _, _, L = self._current_arrays()
        self._A, self._B = self._window_update(X, L, self._A, self._B)

    def is_done(self) -> bool:
        if self._iter == 0: return False
        if self._iter >= self.max_iters: return True
        X, _, _, L = self._current_arrays()
        (x1_lo, x1_hi, have_x1), (x2_lo, x2_hi, have_x2), _ = self._brackets(X, L)
        win = max(1e-12, self._B - self._A)
        ok1 = have_x1 and (x1_lo is not None) and (x1_hi is not None) and ((x1_hi - x1_lo) <= self.tol_frac * win)
        ok2 = have_x2 and (x2_lo is not None) and (x2_hi is not None) and ((x2_hi - x2_lo) <= self.tol_frac * win)
        return bool(ok1 and ok2)

    def get_estimates(self) -> tuple[float | None, float | None]:
        if len(self._X) == 0:
            return None, None
        X, _, _, L = self._current_arrays()
        # first non-L
        if (~(np.array(L) == LblL)).any():
            x1_hat = float(X[int(np.argmax(np.array(L) != LblL))])
        else:
            x1_hat = None
        # first R to the right of x1_hat
        x2_hat = None
        if x1_hat is not None:
            idx = np.where((X >= x1_hat) & (np.array(L) == LblR))[0]
            if len(idx) > 0:
                x2_hat = float(X[idx[0]])
        return x1_hat, x2_hat

    # ---------- internals ----------
    def _pass2(self, f1: float, f2: float) -> bool:
        m = self.x2_mode
        if   m == "min_ge": return min(f1, f2) >= self.r2
        elif m == "min_le": return min(f1, f2) <= self.r2
        elif m == "y1_ge":  return f1 >= self.r2
        elif m == "y1_le":  return f1 <= self.r2
        elif m == "y2_ge":  return f2 >= self.r2
        elif m == "y2_le":  return f2 <= self.r2
        else: raise ValueError(f"Unknown x2_mode: {m}")

    def _label(self, f1: float, f2: float) -> int:
        if max(f1, f2) < self.r1: return LblL
        if self._pass2(f1, f2):   return LblR
        return LblM

    def _current_arrays(self):
        X = np.array(self._X); F1 = np.array(self._F1); F2 = np.array(self._F2); L = np.array(self._L)
        if len(X) == 0: return X, F1, F2, L
        order = np.argsort(X)
        return X[order], F1[order], F2[order], L[order]
    
    def _brackets(self, X, L):
        # ---- x1: first non-L and last L strictly to its left ----
        idx_notL = np.where(L != LblL)[0]
        if len(idx_notL) == 0:
            return (None, None, False), (None, None, False), None

        i1 = idx_notL[0]
        x1_hi = X[i1]
        left_L_idx = np.where((L == LblL) & (np.arange(len(X)) < i1))[0]
        have_x1 = len(left_L_idx) > 0
        x1_lo = X[left_L_idx[-1]] if have_x1 else None
        x1_hat = x1_hi

        # ---- x2: first R to the right of x1_hat and last not-R before it ----
        idx_R = np.where((np.arange(len(X)) >= i1) & (L == LblR))[0]
        if len(idx_R) == 0:
            return (x1_lo, x1_hi, have_x1), (None, None, False), x1_hat

        i2 = idx_R[0]
        x2_hi = X[i2]
        left_notR_idx = np.where((np.arange(len(X)) >= i1) &
                                (np.arange(len(X)) <  i2) &
                                (L != LblR))[0]
        have_x2 = len(left_notR_idx) > 0
        x2_lo = X[left_notR_idx[-1]] if have_x2 else None

        return (x1_lo, x1_hi, have_x1), (x2_lo, x2_hi, have_x2), x1_hat


    def _choose_batch(self, A, B, N, x1_br, x2_br):
        x1_lo, x1_hi, have_x1 = x1_br
        x2_lo, x2_hi, have_x2 = x2_br
        X, _, _, L = self._current_arrays()
        w = max(1e-12, B - A)

        xs = []

        # --- edge guards: force exploration where labels are unseen globally
        haveL = (L == LblL).any()
        haveR = (L == LblR).any()
        if not haveL:
            xs.append(A); xs.append(min(A + 0.1*w, B))
        if not haveR:
            xs.append(B); xs.append(max(B - 0.1*w, A))

        # --- exploitation midpoints (if room left)
        if len(xs) < N:
            xs.append(0.5*(x1_lo + x1_hi) if (have_x1 and x1_lo is not None and x1_hi is not None) else (A+B)/2)
        if len(xs) < N:
            if have_x2 and (x2_lo is not None) and (x2_hi is not None):
                xs.append(0.5*(x2_lo + x2_hi))
            else:
                anchor = (x1_hi + 0.25*w) if (have_x1 and x1_hi is not None) else (A+B)/2
                xs.append(min(max(anchor, A), B))

        # --- frontier segments (between label changes)
        segs: List[Tuple[float,float,float]] = []
        if len(X) >= 2:
            for i in range(len(X)-1):
                a, b = X[i], X[i+1]
                if b <= A or a >= B: continue
                a, b = max(a, A), min(b, B)
                if a >= b: continue
                if L[i] != L[i+1]:
                    bonus = 1.5 if ((L[i] in (LblL, LblM)) and (L[i+1] in (LblM, LblR))) else 1.0
                    segs.append((a, b, (b-a)*bonus))
        if not segs:
            segs = [(A, B, B - A)]

        # avoid deep-R if we have x2 bracket
        if have_x2 and (x2_lo is not None) and (x2_hi is not None):
            right_cap = x2_hi + 2*max(1e-12, x2_hi - x2_lo)
            segs = [(a, min(b, right_cap), w_) for (a,b,w_) in segs if a < min(b, right_cap)]
            if not segs:
                segs = [(A, min(B, right_cap), min(B, right_cap)-A)]

        # allocate remaining slots
        remain = max(0, N - len(xs))
        weights = np.array([w_ for *_, w_ in segs], float)
        weights = weights / max(1e-12, weights.sum())
        alloc = np.random.multinomial(remain, weights)
        for (a, b, _), k in zip(segs, alloc):
            if k <= 0: continue
            base = (np.arange(k)+0.5)/k
            pts  = a + base*(b-a)
            jitter = self.rng.uniform(-1/6, 1/6, size=k)*(b-a)/max(k,1)
            xs.extend((pts + jitter).tolist())

        xs = np.array(xs[:N], float)
        xs = xs[(xs >= A) & (xs <= B)]
        if len(xs) < N:
            xs = np.concatenate([xs, self.rng.uniform(A, B, size=N-len(xs))])
        xs = np.sort(xs)
        return xs

    def _window_update(self, X, L, A, B):
        w = (B - A)
        if len(X) == 0: return A, B
        Lmask = (L == LblL); Rmask = (L == LblR)

        # pure batches unchanged ...
        if Lmask.all():
            self._allL_run += 1; self._allR_run = 0
            step = w * (2 ** max(0, self._allL_run - 1))
            return B, B + step
        if Rmask.all():
            self._allR_run += 1; self._allL_run = 0
            step = w * (2 ** max(0, self._allR_run - 1))
            return A - step, A

        # ---- mixed: use correctly-sided evidence ----
        self._allL_run = self._allR_run = 0

        idx_notL = np.where(L != LblL)[0]
        i1 = idx_notL[0]                      # first non-L exists
        x1_hat = X[i1]

        # do we actually have an L to the LEFT of x1_hat?
        have_left_L = ((L == LblL) & (X < x1_hat)).any()
        if not have_left_L:
            # force a leftward slide to find true "left-of-x1" evidence
            return A - w, A

        # first R to the right of x1_hat
        idx_R = np.where((X >= x1_hat) & (L == LblR))[0]
        if len(idx_R) == 0:
            # no R yet → right-heavy search
            return x1_hat - 0.25*w, x1_hat + 1.25*w

        i2 = idx_R[0]
        x2_hat = X[i2]
        have_notR_before_R = ((L != LblR) & (X >= x1_hat) & (X < x2_hat)).any()

        if not have_notR_before_R:
            # R exists but no not-R before it → widen around x1 to create a real bracket
            return x1_hat - 0.25*w, x1_hat + 1.25*w

        # proper brackets for both → tighten with small pad
        pad = 0.25 * max(1e-12, x2_hat - x1_hat)
        return x1_hat - pad, x2_hat + pad
