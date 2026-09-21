# CAID rebuild notes (runall + ProstT5 + README.txt)

## What changed

1. ProstT5 remains bundled in the image (`/app/models/Rostlab-ProstT5`).
2. Default entry is `MODE=runall` via `/app/runall.sh`:
   extract embeddings + predict in one pass.
3. `timings.csv` milliseconds = wall-clock **extract + predict** per sequence.
4. README is `/app/README.txt` (not `.md`).

## Files touched

- `cliper/caid_runall.py` (new)
- `cliper/cli.py` (`runall` subcommand)
- `submission/caid/runall.sh`
- `submission/caid/entrypoint.sh`
- `submission/caid/Dockerfile.full`
- `submission/caid/README-CLIPer-LinkCNN.txt`
- `submission/caid/README-CLIPer-LinkCRNN.txt`
- `scripts/package_caid_release.sh`

## Test after build

```bash
docker run --rm -e NUM_THREADS=4 \
  -v /path/to/targets.fasta:/input/targets.fasta:ro \
  -v /path/to/out:/output \
  cliper-linkcnn:latest

ls /path/to/out/linker/*.caid
head /path/to/out/timings.csv
cat /path/to/out/runall_summary.json | head
docker run --rm --entrypoint cat cliper-linkcnn:latest /app/README.txt
```
