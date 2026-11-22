# Requests

One request per file, `<date>-<package>-<short>.yaml` (date YYYYMMDD), added on the package's own branch and merged
with it. The coordinator answers in `configs/rel/coordinator/` (never by editing the request). Fields:

```yaml
package: W6
kind: gpu_slot          # gpu_slot | cpu_run (a CPU run over 15 minutes) | contract_change | to_package
to: C0                  # C0, P0 or a package
what: two seeds of variant W6-reference in round 1
why: the reference arm of the round
size: {hours: 1.0, threads: 2, disk_gb: 5, gpu_mem_gb: 20}
by: 2026-11-28
```

A GPU slot also needs the variant file. Repository text is public: no internal paths, no paths of a developer machine,
no names of people.
