NEURON {
  POINT_PROCESS BgNoiseClamp
  NONSPECIFIC_CURRENT i
  RANGE mu, sigma, noise
  POINTER pmu, psigma
}

UNITS {
  (nA) = (nanoamp)
}

PARAMETER {
  mu = 0 (nA)
  sigma = 0 (nA)
  noise = 0  : zero-mean noise played from Python (dimensionless)
}

ASSIGNED {
  i (nA)
  pmu    : mean current pointer (nA)
  psigma : current std. pointer (nA)
}

LOCAL s

BREAKPOINT {
  :i = -(pmu + sigma * noise)
  s = sigma + psigma
  if (s < 0) {
      s = 0
  }
  i = -mu + s * noise
}
