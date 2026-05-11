Optional local wheels (air-gapped or exact site mirrors of what went into `structcheck-rl.sqsh`).

- Add `*.whl` here before `docker build`. They install **after** `requirements.txt`.
- With no `.whl` files, the install step is a no-op.

Slurm jobs use the frozen sqsh; this folder only affects **Docker** builds.
