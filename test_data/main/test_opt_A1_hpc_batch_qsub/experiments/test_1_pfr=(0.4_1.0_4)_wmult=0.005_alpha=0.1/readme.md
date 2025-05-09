#### Model:
- Conductance OU input
- Connectivity: 0.5% of the original

#### I-R parameters:
- `ou_std = 0.4 * ou_mean` (without intercept)

### U-C optimization parameters:
- Constant `Rc` step size `alpha=0.1`
- Always update `Ru` to the newest value (equiv. to `alpha=1`)
- Skipping steps by indiividual pops. is not allowed

### Problems:
