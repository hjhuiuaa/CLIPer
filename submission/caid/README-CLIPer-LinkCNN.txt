CLIPer-LinkCNN — Docker Usage
=============================

CLIPer-LinkCNN provides a Docker wrapper that accepts FASTA on standard input and
returns a gzip-compressed tar archive on standard output, so you can run
predictions without a local Python / PyTorch toolchain.

The in-container entry script is `dockerinout`: it writes stdin to a temporary
FASTA, runs extract+predict (ProstT5 + LinkCNN), then tar-gzips CAID outputs
under the working results to stdout.


Prerequisites
-------------

- Docker Engine installed (https://docs.docker.com/engine/install/)
- Network access to pull the image


Option 1: Pull from Aliyun ACR
------------------------------

1. Pull the image:

```bash
docker pull cliper-registry.eu-central-1.cr.aliyuncs.com/cliper/cliper-linkcnn:latest
```

2. Run (keep the `<` and `>` redirection):

```bash
docker run -i cliper-registry.eu-central-1.cr.aliyuncs.com/cliper/cliper-linkcnn:latest \
  dockerinout < test.fasta > cliper_linkcnn_results.tar.gz
```

Where:

- `test.fasta` — input FASTA (2-line records: header + sequence; no label lines)
- `cliper_linkcnn_results.tar.gz` — output archive containing predictions and timings


Option 2: Pull from Docker Hub
------------------------------

1. Pull the image:

```bash
docker pull hjhuiuaa/cliper-linkcnn:latest
```

2. Run (keep the `<` and `>` redirection):

```bash
docker run -i hjhuiuaa/cliper-linkcnn:latest dockerinout < test.fasta > cliper_linkcnn_results.tar.gz
```


Extract and inspect results
---------------------------

```bash
tar -xzf cliper_linkcnn_results.tar.gz
```

Typical files:

| Path | Description |
|------|-------------|
| `linker/{id}.caid` | Per-sequence predictions |
| `timings.csv` | Per-sequence runtime (milliseconds = extract + predict) |

`.caid` format:

```text
>P04637
1	M	0.892	1
2	E	0.813	1
...
```

`timings.csv` format:

```text
# Running CLIPer linker predictor, started ...
sequence,milliseconds
P04637,1827
```


The container `ENTRYPOINT` is `/bin/bash`, so you must pass `dockerinout`
explicitly when running.


Dependencies (included in the image)
------------------------------------

The image bundles ProstT5 weights/tokenizer, the LinkCNN checkpoint, and the
Python runtime used by `dockerinout` / `runall`. The host does not need PyTorch
or transformers.


Example
-------

```bash
cat > smoke.fasta <<'EOF'
>seq1
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL
EOF

docker run -i hjhuiuaa/cliper-linkcnn:latest dockerinout < smoke.fasta > smoke_results.tar.gz
mkdir -p smoke_out && tar -xzf smoke_results.tar.gz -C smoke_out
ls smoke_out/linker/
cat smoke_out/timings.csv
head smoke_out/linker/seq1.caid
```
