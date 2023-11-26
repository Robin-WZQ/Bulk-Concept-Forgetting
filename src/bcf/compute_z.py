from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

from src.bcf import repr_tools
from src.util import nethook

from .bcf_hparams import BCFHyperParams

def get_module_input_output_at_words(
    model: CLIPModel,
    tok: CLIPTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()

def get_lookup_idxs(prompts, subject, processor, hparams):
    if subject:
        lookup_idxs = [
            find_fact_lookup_idx(
                prompt, subject, processor.tokenizer, hparams.fact_token
            )
            for i, prompt in enumerate(prompts)
        ]
    else:
        lookup_idxs = [len(processor.tokenizer(prompt)["input_ids"]) - 2  for prompt in prompts]
    return lookup_idxs

def compute_z(
    model: CLIPModel,
    processor: CLIPProcessor,
    request: Dict,
    hparams: BCFHyperParams,
    layer: int,
    context_templates: List[str],
    clean_model,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runs a simple optimization procedure.
    """

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts_i = [request["prompt"].replace(request["subject"], '{}')] + context_templates
    rewriting_prompts_o = list(set(rewriting_prompts_i))
    rewriting_prompts_o.sort(key = rewriting_prompts_i.index)

    # Compute indices of the tokens where the fact is looked up
    context_text_image_lookup_idxs = get_lookup_idxs(rewriting_prompts_o, request["subject"], processor, hparams)
    
    rewriting_prompts = [p.format(request["subject"],) for p in rewriting_prompts_o]    

    # Compute re-write inputs
    target_id = 0
    if request["algorithm"] == "bcf":
        context_text_text_prompts = rewriting_prompts
        target_text_text_prompts = [ request["new_text"], request["true_text"]] + request.get("kl_prompts", [])
        # target_text_text_prompts = [p.format(request["new_text"],) for p in rewriting_prompts_o]  + request.get("kl_prompts", [])
        context_text_text_lookup_idxs = context_text_image_lookup_idxs
    else:
        raise Exception(f"Unknown algorithm: {request['algorithm']}")

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.text_model.config.hidden_size,), requires_grad=True, device="cuda")
    target_init = None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                # print("\nRecording initial value of z*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0], :].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)


    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        context_text_text_scores = None

        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:

            if request["algorithm"] == 'bcf':
                context_input = processor(text=context_text_text_prompts, return_tensors="pt", padding=True).to("cuda")
                target_input = processor(text=target_text_text_prompts, return_tensors="pt", padding=True).to("cuda")
                lookup_idxs = context_text_text_lookup_idxs

                device = model.device
                context_embeddings = model.get_text_features(**context_input)
                model = model.to("cpu")
                
                clean_model = clean_model.to(device)
                with torch.no_grad():
                    target_embeddings = clean_model.get_text_features(**target_input)
                clean_model.to("cpu")
                model.to(device)
                
                if request["similarity_metric"] == "cosine":
                    logit_scale = model.logit_scale.exp()
                    context_embeddings = context_embeddings / context_embeddings.norm(p=2, dim=-1, keepdim=True)
                    target_embeddings = target_embeddings / target_embeddings.norm(p=2, dim=-1, keepdim=True)
                    context_text_text_scores = torch.matmul(context_embeddings, target_embeddings.t()) * logit_scale

                elif request["similarity_metric"] == "l2":
                    logits_per_text = - torch.cdist(
						torch.unsqueeze(context_embeddings, dim=0),
						torch.unsqueeze(target_embeddings, dim=0)
					)
                    context_text_text_scores = torch.squeeze(logits_per_text)
                    
        # Compute loss on rewriting targets
        if request["algorithm"] == "bcf":
            context_text_text_log_probs = torch.log_softmax(context_text_text_scores, dim=1)
            loss = context_text_text_log_probs[:, target_id]
            avg_value = torch.exp(context_text_text_log_probs[:, target_id]).mean(0).item()
        else:
            raise Exception(f"Unknown algorithm: {request['algorithm']}")

        loss = loss.mean(0)
        nll_loss = -loss

        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        
        loss = nll_loss + weight_decay
        
        if "bcf" in request["algorithm"] and avg_value > hparams.v_prob_threshold:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    return target


def get_module_input_output_at_word(
    model: CLIPTextModel,
    tok: CLIPTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )

    subtoken = fact_token_strategy[len("subject_") :]
    l_input, l_output = repr_tools.get_reprs_at_word_tokens(
        track="both",
        subtoken=subtoken,
        context_templates=context_template,
        words=word,
        **word_repr_args,
    )

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: CLIPTokenizer,
    fact_token_strategy: str,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return ret
