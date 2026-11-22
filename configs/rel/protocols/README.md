# Protocols

One file per round, `<round>.yaml`, written by the programme lead and committed before the round's deadline: the
coverage (which groups and scenarios, what is not covered), metrics, the effect sizes that count, the interval method,
the shares of the masking schemes, and the slots (variant, seed, budget, machine). The coordinator writes the round's
profiles from it. Its block `tolerance` holds the thresholds `python -m ci compare A B --expect within_tolerance
--protocol <file>` passes to the tolerance function (after round 1).
