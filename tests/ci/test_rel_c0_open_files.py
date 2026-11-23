"""The table writer opens every shard of a part at once (one memmap, so one open file, per shard and field): it raises
the soft limit of open files to hold them (v0_io.open_files_for), so a part of many shards does not fail with "too many
open files" under a soft limit of 1,024 (the version-1 build of 201,600 tables in shards of 2,400 did)."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import resource, sys
sys.path.insert(0, sys.argv[1])
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
from seeddrift.train.rel.tables import v0_io as IO
nf = len(IO.fields(4, 1))
shards, mm = IO.create(sys.argv[2], 40, 4, 1, 1)
assert len(shards) == 40 and len(mm) == 40 * nf > 64, (len(mm), nf)
for x in mm.values():
    x.flush()
now = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
assert now >= 40 * nf, now
print("ok", len(mm), now)
"""


def test_create_raises_the_soft_limit_for_many_shards():
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([sys.executable, "-c", CHILD, ROOT, os.path.join(d, "part")], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.startswith("ok ")


def test_open_files_for_never_lowers_the_limit():
    import resource

    from seeddrift.train.rel.tables import v0_io as IO
    before = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert IO.open_files_for(0, spare=0) == before[0]
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == before
