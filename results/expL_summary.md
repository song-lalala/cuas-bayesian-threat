# Exp-L: computational cost

- machine: AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD, Python 3.14.4, NumPy 2.4.4; single core, no GPU
- replications: 5; queries per replication: 2500

- offline grounding (supervised, N=200): 366 ms
- offline grounding (MAP-EM, 12 iterations, latent layer): 1590 ms (4x supervised)
- deployment assembly (3 range bins): 2.2 ms
- per-track inference: mean 0.106 ms, median 0.104 ms, p95 0.126 ms, max 0.403 ms
- sustained single-core throughput: 9475 track updates/s
