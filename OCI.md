# Nahual ViT OCI image

`nix build .#oci-image` creates `nahual/vit:local`, loadable with either
`podman load < result` or `docker load < result`. One image serves both wrapped
models. Its first argument selects `morphem` (default) or `openphenom`; the
second optionally overrides `tcp://0.0.0.0:5555`.

```console
podman run --rm --device nvidia.com/gpu=all -p 5555:5555 \
  -v nahual-vit-cache:/tmp/nahual nahual/vit:local morphem
# Or: ... nahual/vit:local openphenom
```

Use Docker's `--gpus all`; both variants fall back to CPU. The volume persists
Hugging Face code and weights.

```console
pip install 'nahual==0.0.8' numpy
NAHUAL_VIT_VARIANT=morphem python oci/smoke_test.py
```

Run the test once against each server variant. It downloads the corresponding
pretrained model, sends NCZYX input over TCP, and validates the embedding.
