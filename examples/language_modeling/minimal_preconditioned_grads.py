import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from logix import LogIX
from logix.config import LoggingConfig

MODEL_NAME = "Qwen/Qwen3.5-2B"
PROJECT = "qwen3_5_2b_minimal_preconditioned_grads"
NAME_FILTER = ["19.mlp.up_proj"]
NUM_CALIBRATION_SAMPLES = 32
MAX_LENGTH = 128
DAMPING = 1e-3
QUERY_TEXT = "The Eiffel Tower is located in"


def encode(tokenizer, text, device):
    return {
        key: value.to(device)
        for key, value in tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).items()
    }


def causal_lm_loss(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
    )
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = batch["input_ids"][..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="sum",
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model_kwargs = {}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs).to(device)
    model.eval()

    run = LogIX(
        project=PROJECT,
        logging_config=LoggingConfig(log_dtype="float32"),
    )
    run.watch(
        model,
        type_filter=[nn.Linear],
        name_filter=NAME_FILTER,
    )

    calibration_dataset = load_dataset(
        "openwebtext",
        split="train",
        streaming=True,
    )

    run.setup({"forward": ["covariance"], "backward": ["covariance"]})
    for idx, example in enumerate(calibration_dataset):
        if idx == NUM_CALIBRATION_SAMPLES:
            break
        batch = encode(tokenizer, example["text"], device)
        print(idx, tokenizer.decode(batch["input_ids"])[0])
        print(idx, "got batch")
        with run(data_id=[f"calibration_{idx}"], mask=batch.get("attention_mask")):
            model.zero_grad(set_to_none=True)
            loss = causal_lm_loss(model, batch)
            loss.backward()
    run.finalize()

    query_batch = encode(tokenizer, QUERY_TEXT, device)
    run.eval()
    run.setup({"grad": ["log"]})
    with run(data_id=["query"], mask=query_batch.get("attention_mask")) as session:
        model.zero_grad(set_to_none=True)
        loss = causal_lm_loss(model, query_batch)
        loss.backward()
        query_log = session.get_log(copy=True)

    preconditioned_log = run.influence.precondition(
        query_log,
        hessian="kfac",
        damping=DAMPING,
    )

    _, preconditioned_grads = preconditioned_log
    for module_name, module_log in preconditioned_grads.items():
        grad = module_log["grad"]
        assert torch.isfinite(grad).all(), module_name
        print(module_name, tuple(grad.shape), grad.norm().item())

    breakpoint()


if __name__ == "__main__":
    main()
