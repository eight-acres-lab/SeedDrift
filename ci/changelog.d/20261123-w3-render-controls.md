# W3: the renderer's two controls spare row 0, not a block's frame row (renderer version 2)

- `render(..., control="shuffle" | "presence")` acts on every row but row 0 (the examined row). Version 1 spared a
  block's frame row, which is row 0 except for a target block laid on its frame relative (whole gene hidden): there the
  presence control left the frame relative's codons visible and the shuffle moved row 0 together with the other rows,
  keeping its correspondence with them. Blocks laid on row 0 render as before.
