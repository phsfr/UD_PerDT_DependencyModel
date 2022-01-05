#!/bin/sh

[ $# -ge 1 ] || { echo Usage: $0 data_directory embedding_args... >&2; exit 1; }
data="$1"; shift

for d in $data/*/; do
  for f in $d*.conllu; do
    [ $f.npz -nt $f ] && continue
    wembedding_service/venv/bin/python.exe wembedding_service/compute_wembeddings.py --format=conllu $f $f.npz "$@"
  done
done
