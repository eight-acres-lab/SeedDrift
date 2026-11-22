"""Models: one architecture per module (models/<family>.py), one author each; models/common/ holds the shared
layers (W6). A model module exposes logp(rendered, target, truth=None) and info (the contract, api.py) and may import
torch at the top. uniform.py is the stub arm of the coordinator (numpy only)."""
