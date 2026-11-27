"""Read a multi-shard set in a child with the usual low descriptor limit."""
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHILD = r'''
import hashlib, os, resource, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from seeddrift.train.rel.evalh.shards import TableSet

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
folder = sys.argv[2]
os.makedirs(folder)
files = {}
for i in range(160):
    name = 's%04d' % i
    arrays = dict(n=np.full((2, 1, 1), i + 1, np.int16),
                  split=np.zeros(2, np.int8), group=np.full(2, i % 12, np.int8),
                  tok=np.arange(i * 6, i * 6 + 6, dtype=np.int32).reshape(2, 1, 3))
    for field, value in arrays.items():
        path = os.path.join(folder, name + '.' + field + '.npy')
        np.save(path, value)
        with open(path, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        files[name, field] = dict(path=os.path.basename(path), bytes=os.path.getsize(path), sha256=digest)
tables = TableSet(folder, '.', files)
assert tables.tables == 320 and tables.split() == 0
for i in [159, 0, 79, 12, 158] + list(range(160)):
    got = tables.load(i * 2, i * 2 + 2, ('n', 'group', 'tok'))
    assert np.array_equal(got['tok'], np.arange(i * 6, i * 6 + 6, dtype=np.int32).reshape(2, 1, 3))
    assert (got['n'] == i + 1).all() and (got['group'] == i % 12).all()
assert len(tables._open) == 640
assert len(tables.verified) == 160 * 3
assert resource.getrlimit(resource.RLIMIT_NOFILE)[1] == hard
print('ok', tables.tables, len(tables._open), resource.getrlimit(resource.RLIMIT_NOFILE)[0])
'''


def test_reader_raises_its_soft_limit_before_mapping_many_shards(tmp_path):
    out = subprocess.run([sys.executable, "-c", CHILD, ROOT, str(tmp_path / "tables")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.startswith("ok 320 640 ")


MULTI = r'''
import hashlib, os, resource, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from seeddrift.train.rel.evalh.shards import TableSet
from seeddrift.train.rel.tables import v0_io as IO

_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
sets = []
for part, count in [('train', 168), ('heldout', 2), ('scattered', 2), ('suffix', 2), ('whole_gene', 2)]:
    folder = os.path.join(sys.argv[2], part)
    os.makedirs(folder)
    files = {}
    for i in range(count):
        shard = 's%04d' % i
        arrays = IO.alloc(1, 1, version=1)
        arrays['group'][:] = i % 12
        arrays['split'][:] = 0 if part == 'train' else 1
        arrays['gene'][:, -1] = i + 123
        for field, value in arrays.items():
            path = os.path.join(folder, shard + '.' + field + '.npy')
            np.save(path, value)
            with open(path, 'rb') as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            files[shard, field] = dict(path=os.path.basename(path), bytes=os.path.getsize(path), sha256=digest)
    tables = TableSet(folder, '.', files)
    assert tables.split() == (0 if part == 'train' else 1)
    sets.append(tables)
for tables in sets:
    for i in range(tables.tables - 1, -1, -1):
        got = tables.load(i, i + 1, tables.fields)
        assert got['gene'][0, -1] == i + 123
        assert got['group'][0] == i % 12
    assert len(tables._open) == len(tables.files)
    assert len(tables.verified) == len(tables.files)
assert resource.getrlimit(resource.RLIMIT_NOFILE)[1] == hard
assert sum(len(t._open) for t in sets) == 176 * len(IO.fields(1, version=1))
print('ok multi', sum(len(t._open) for t in sets))
'''


def test_expanded_train_and_all_heldout_parts_fit_together(tmp_path):
    out = subprocess.run([sys.executable, "-c", MULTI, ROOT, str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.startswith("ok multi 3168")
