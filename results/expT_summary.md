# Exp-T: tornado sensitivity of P(T=high | e*) (worked-track evidence; ESS=2.0)

- baseline P(T=high|e*) = 0.633
- parameters examined: 183 (all learned internal CPT entries; +-0.1 proportional co-variation)
- linear-fractional fit succeeded for 128/183 entries

## top-15 by posterior swing
- T(2, 1, 2)k3: theta=0.852, P in [0.565, 0.700] (swing +0.136)
- T(2, 1, 2)k2: theta=0.146, P in [0.565, 0.700] (swing -0.135)
- C(1, 1)k2: theta=0.756, P in [0.577, 0.688] (swing +0.111)
- C(1, 1)k1: theta=0.244, P in [0.577, 0.688] (swing -0.111)
- C(1, 1)k0: theta=0.000, P in [0.571, 0.632] (swing -0.061)
- H(2, 2, 1)k0: theta=0.000, P in [0.571, 0.632] (swing -0.061)
- T(2, 1, 2)k1: theta=0.001, P in [0.575, 0.633] (swing -0.058)
- T(2, 1, 2)k0: theta=0.000, P in [0.575, 0.632] (swing -0.057)
- N(2, 1)k1: theta=1.000, P in [0.578, 0.632] (swing +0.054)
- N(2, 1)k0: theta=0.000, P in [0.578, 0.632] (swing -0.054)
- T(2, 1, 1)k3: theta=0.237, P in [0.611, 0.654] (swing +0.044)
- T(2, 1, 1)k2: theta=0.742, P in [0.613, 0.653] (swing -0.040)
- H(2, 2, 1)k2: theta=1.000, P in [0.603, 0.632] (swing +0.029)
- H(2, 2, 1)k1: theta=0.000, P in [0.603, 0.632] (swing -0.029)
- N(2, 0)k0: theta=0.477, P in [0.622, 0.643] (swing -0.020)

## credal bounds (top-5, one-at-a-time +-0.1)
- one-at-a-time envelope over the top-5 parameters: P(T=high|e*) in [0.565, 0.700] (baseline 0.633) -- the threat decision is invariant over the credal set at the 0.5 decision level
