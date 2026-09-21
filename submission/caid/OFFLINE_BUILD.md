# Offline / blocked-network Docker build

When the server cannot reach `registry-1.docker.io` and you have **no sudo** to edit `/etc/docker/daemon.json`, use one of these.

---

## Method A (recommended): Mirror in Dockerfile — no sudo

`Dockerfile.full` already defaults to:

```dockerfile
FROM docker.m.daocloud.io/library/python:3.11-slim-bookworm
```

Test pull on the server:

```bash
docker pull docker.m.daocloud.io/library/python:3.11-slim-bookworm
```

If that works:

```bash
./scripts/package_caid_bundle.sh stage4
```

If DaoCloud fails, try another mirror in `docker build`:

```bash
docker build -f submission/caid/Dockerfile.full \
  --build-arg BASE_IMAGE=docker.1ms.run/library/python:3.11-slim-bookworm \
  -t cliper-caid-full:linker-stage4-cnn-t1 .
```

Other mirrors to try (availability varies):

| Mirror | BASE_IMAGE |
|--------|------------|
| DaoCloud | `docker.m.daocloud.io/library/python:3.11-slim-bookworm` |
| 1ms | `docker.1ms.run/library/python:3.11-slim-bookworm` |
| Aliyun | `registry.cn-hangzhou.aliyuncs.com/library/python:3.11-slim-bookworm` |

---

## Method B: Import base image from your PC (offline)

### On Windows (Docker Desktop installed, can access Docker Hub)

```powershell
docker pull python:3.11-slim-bookworm
docker save python:3.11-slim-bookworm -o python-3.11-slim-bookworm.tar
scp python-3.11-slim-bookworm.tar clipersrv:/data/huggs/hujh/CLIPer/
```

### On server

```bash
cd /data/huggs/hujh/CLIPer
docker load -i python-3.11-slim-bookworm.tar
docker images | grep python

# Build using local image (no Hub pull)
docker build -f submission/caid/Dockerfile.full \
  --build-arg BASE_IMAGE=python:3.11-slim-bookworm \
  -t cliper-caid-full:linker-stage4-cnn-t1 .
```

Or after `package_caid_bundle.sh` copies checkpoint:

```bash
cp artifacts/runs/stage4_prostt5_cnn_concat_window_t1/exp0001/checkpoints/best.pt submission/caid/models/best.pt
docker build -f submission/caid/Dockerfile.full \
  --build-arg BASE_IMAGE=python:3.11-slim-bookworm \
  -t cliper-caid-full:linker-stage4-cnn-t1 .
```

---

## Method C: Ask lab admin (needs sudo)

Send admin this `daemon.json`:

```json
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://hub-mirror.c.163.com"
  ]
}
```

Path: `/etc/docker/daemon.json`, then `sudo systemctl restart docker`.

---

## pip timeout during build

Dockerfile uses Tsinghua PyPI by default. Override:

```bash
docker build ... --build-arg PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
```

---

## Build context ~16GB

If `models/Rostlab-ProstT5` is ~11GB on disk, **16GB context is expected** (model + code + checkpoint). This is not the network error.
