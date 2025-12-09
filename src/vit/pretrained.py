"""Model deployment for generic transformers."""

import sys
from functools import partial

import numpy
import pynng
import torch
import transformers
import trio
from loguru import logger
from nahual.server import responder
from transformers import AutoModel

# address = "ipc:///tmp/openphenom.ipc"
address = sys.argv[1]


logger.add(address.split("/")[-1])

shape_guardrails = {
    "recursionpharma/OpenPhenom": (6, 256, 256),
}


def setup(model_name: str, **kwargs) -> dict:
    # Some default values
    device_id = kwargs.get("device", 0)

    setup_defaults = dict(
        device=torch.device(device_id),
    )
    execution_defaults = dict()

    setup_kwargs = kwargs.get("setup_kwargs", {})
    execution_kwargs = kwargs.get("setup_kwargs", {})

    # Fill kwargs with default
    # for k, v in setup_defaults.items():
    #     setup_defaults[k] = setup_kwargs.pop(kwargs, v)

    # for k, v in execution_defaults.items():
    #     execution_defaults[k] = execution_kwargs.pop(kwargs, v)

    # Define parameters by combining defaults and non-defaults
    setup_params = {**setup_defaults, **setup_kwargs}
    execution_params = {**execution_defaults, **execution_kwargs}

    use_gpu = setup_params.pop("gpu", True)
    # Load model instance
    model = AutoModel.from_pretrained(
        "recursionpharma/OpenPhenom",
        trust_remote_code=True,
        dtype="auto",
        **setup_defaults,
    ).cuda()

    execution_params["shape_guardrails"] = shape_guardrails[model_name]

    # Generate a json-encodable dictionary to send back to the client
    serializable_params = {
        name: {k: str(v) for k, v in d.items()}
        for name, d in zip(("setup", "execution"), (setup_params, execution_params))
    }

    # "Freeze" model in-place
    processor = partial(process_pixels, model=model, **execution_params)
    return processor, serializable_params


async def main():
    """Main function for the asynchronous server.

    This function sets up a nng connection using pynng and starts a nursery to handle
    incoming requests asynchronously.

    Parameters
    ----------
    address : str
        The network address to listen on.

    Returns
    -------
    None
    """

    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"Pretrained ViT server listening on {address}")
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


def process_pixels(
    pixels: numpy.ndarray, model: transformers.modeling_utils.PreTrainedModel, **kwargs
) -> numpy.ndarray:
    """Apply a pretrained model"""

    shape_guardrail = kwargs.pop("shape_guardrail")
    assert pixels.shape[-3:] == shape_guardrail, (
        f"Invalid input shape {pixels.shape}. Last dims should be {shape_guardrail}."
    )
    pixels_torch = torch.from_numpy(pixels).float().cuda()
    # TODO make this general, for now the output is fixed
    embeddings, _, _ = model(pixels_torch, **kwargs)  # embeddings, reconstruction, mask
    # [torch.Size([1, 1537, 384]), torch.Size([1, 1536, 256]), torch.Size([1, 1536])]

    embeddings_np = embeddings.cpu().detach().numpy()

    return embeddings_np


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        # that's the way the program *should* end
        pass
