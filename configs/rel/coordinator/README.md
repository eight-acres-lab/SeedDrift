# Coordinator state

Written by the coordinator (package C0) only, so that any session can take the role over:

- `state.yaml`: the integration branch, the commit every package branches from, the branches merged (commit, checks),
  the runs started (run id, machine, profile, commit, status), the queued requests, the tables copied between machines.
- `calendar.yaml`: the lock calendar (each machine's gpu and select lock, by day and hour): the rounds' long and short
  trainings, the CPU windows of the packages, the rules it follows. Ask for a slot with a request file.

Packages branch from the commit named there (or a later tip of `exp/rel`) as `exp/rel-<package>` and merge only
`exp/rel`, never each other's branches.
