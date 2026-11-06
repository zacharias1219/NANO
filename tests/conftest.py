import os

# Parity is checked single-threaded so BLAS reduction order is deterministic (PRD non-functional reqs).
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
