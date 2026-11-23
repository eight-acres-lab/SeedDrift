# Coordinator: a training run of several arms passes only with every arm

- Gate G4_rel_w6_fit fails a run in which an arm was not trained (refused for GPU memory, failed its preflight or its
  training) or ended before its budget (`w6_fit.arm_problems`: its budget_cells where it has one, max_steps where the
  profile sets it, otherwise the wall limit); a preflight-only run fails when an arm was not preflighted. The evidence
  names each arm and why. Before, one trained arm was enough and the others were only recorded: run 20261123T110619
  passed with its reference arm refused for memory. The run's files stay for reading; it hands nothing off.
