# Coordinator: the table writer holds every shard of a part open

- `seeddrift/train/rel/tables/v0_io.create` raises the process's soft limit of open files to hold every shard and field
  of the part it opens (each writable memmap keeps its file open), never above the hard limit
  (`v0_io.open_files_for`). The version-1 build of 201,600 tables in shards of 2,400 failed with "too many open files"
  under a soft limit of 1,024. What is written is unchanged.
