# Optional Docker environment

The supported default installation is the root-level `environment.yml`.
Docker is optional and mirrors the same pinned `requirements.txt` on the
PyTorch 2.7.0 / CUDA 11.8 base image.

Build from the repository root:

```bash
docker build -f docker/Dockerfile -t preserve-the-hard .
```

Run with the repository and a dataset directory mounted explicitly:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v "$PWD:/workspace/preserve_the_hard" \
  -v "/path/to/datasets:/datasets:ro" \
  -w /workspace/preserve_the_hard \
  preserve-the-hard
```

No host-specific paths, credentials, or model caches are embedded in the
image. Pass any optional Hugging Face or W&B credentials at runtime.
