# W18: the readout refuses a key written twice

- rel_w18_readout stops on a per-cell file that scores a (table, pass, slot) twice or names a pass twice (the presence
  passes are told apart from the plain ones by their pass names), and its gate fails when stats.tsv, criterion.tsv,
  the verdicts, the second verdicts, the arms or the columns of scores.tsv hold one key twice (after the programme
  lead's note of 2026-11-24 on a report that let a control's rows overwrite the plain reading's).
