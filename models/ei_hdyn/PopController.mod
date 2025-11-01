NEURON {
  POINT_PROCESS PopController
  RANGE tau, r0, k, rate, z, z0
}

PARAMETER {
  tau = 50 (ms)         : EMA time constant for rate
  r0  = 5               : target rate (Hz)
  k   = 0.02            : adaptation gain
  z0 = 0                : initial value of z
}

STATE {
  rate                  : (Hz)
  z                     : control variable
}

INITIAL {
  rate = 0
  z = z0
}

BREAKPOINT {
  SOLVE states METHOD cnexp
}

DERIVATIVE states {
  rate' = -rate/tau
  z'   = k*(r0 - rate)
}

NET_RECEIVE (w) {
  : Each spike bumps the EMA by 1000/tau (so steady-state == spike freq in Hz)
  rate = rate + w * (1000.0/tau)
}
