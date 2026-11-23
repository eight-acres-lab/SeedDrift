# W6: the training stage on a shared GPU

- `rel_w6_fit` measures the free GPU memory again when each arm is to start (the least of that and what was free when
  the trainings began, less what the running arms need): another user's process may grow between the preflights and
  the trainings (run 20261123T110619). An arm that no longer fits waits while one of its run's arms trains, and is
  otherwise refused (the run then fails at its gate).
- `rel_w6_fit.micro_fallback` (default false): such an arm is preflighted again within what is free then, its
  micro-batch halved until it fits (tables per step unchanged); the first preflight is kept as `preflight_first.json`,
  the index records the fallback (`info.micro_fallback`) and the rows the micro-batch trained with.
