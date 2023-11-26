from typing import Dict, List, Tuple

import torch
from transformers import CLIPProcessor, CLIPModel

from src.bcf import apply_bcf_to_model

from src.util import nethook
from src.util.globals import *


def demo_model_editing(
    model: CLIPModel,
    processor: CLIPProcessor,
    requests: List[Dict],
    concept_type: str,
    alg_name: str = "BCF",
    hparams=None,
) -> Tuple[CLIPModel, Dict[str, torch.Tensor]]:
    """
    Applies the selected model editing algorithm. 
    Returns the updated model and the original values of weights that were changed.
    """

    nethook.set_requires_grad(True, model)

    apply_method = apply_bcf_to_model

    print_loud(f"Applying {alg_name} to model")
    model_new, orig_weights = apply_method(
        model,
        processor,
        requests,
        hparams,
        concept_type,
        return_orig_weights=True,
    )

    return model_new, orig_weights


def print_loud(x, pad=3):
    """
    Prints a string with # box for emphasis.

    Example:
    ############################
    #                          #
    #  Applying BCF to model  #
    #                          #
    ############################
    """

    n = len(x)
    print()
    print("".join(["#" for _ in range(n + 2 * pad)]))
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print(
        "#"
        + "".join([" " for _ in range(pad - 1)])
        + x
        + "".join([" " for _ in range(pad - 1)])
        + "#"
    )
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print("".join(["#" for _ in range(n + 2 * pad)]))


class StopExecution(Exception):
    def _render_traceback_(self):
        pass


def stop_execution():
    raise StopExecution
