"""Stages of the relational-model line (pipelines rel_*; configs/rel/kinds.yaml). One module per stage, named
<package>_<what>.py, owned by its package; kinds.py, common.py, gates.py, summary.py, stub.py, synth.py and c0_synth.py
belong to the coordinator (C0). Nothing is imported here: a train-side stage must not pull in the generator of
synthetic tables (synth.py) by importing this package."""
