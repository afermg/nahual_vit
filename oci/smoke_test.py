#!/usr/bin/env python3
"""End-to-end inference against a Nahual ViT OCI variant."""

import json
import os

os.environ.setdefault("NAHUAL_IPC_TIMEOUT_MS", "900000")

import numpy as np
from nahual.process import dispatch_setup_process


def main() -> None:
    address = os.environ.get("NAHUAL_ADDRESS", "tcp://127.0.0.1:5555")
    variant = os.environ.get("NAHUAL_VIT_VARIANT", "morphem").lower()
    if variant == "morphem":
        model_name = "CaicedoLab/MorphEm"
        channels = 2
        expected_features = 768
    elif variant == "openphenom":
        model_name = "recursionpharma/OpenPhenom"
        channels = 5
        expected_features = 384
    else:
        raise ValueError(f"unknown variant: {variant}")

    setup, process = dispatch_setup_process("vit")
    info = setup({"model_name": model_name, "device": "cpu"}, address=address)
    pixels = np.random.default_rng(42).random(
        (1, channels, 1, 224, 224), dtype=np.float32
    )
    result = process(pixels, address=address)
    assert result.shape == (1, expected_features), result.shape
    assert np.isfinite(result).all()
    print(json.dumps({"variant": variant, "setup": info, "shape": list(result.shape)}))


if __name__ == "__main__":
    main()
