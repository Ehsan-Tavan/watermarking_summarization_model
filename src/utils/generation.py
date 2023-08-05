# ============================ Third Party libs ============================
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding


def load_model(path: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    model.to(device)
    model.eval()
    return model, tokenizer


def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None):
    """collate batch of input_ids into a padded batch of tensors"""
    assert (input_ids[0].shape[0] == 1 and input_ids[0].shape[1] > 0), \
        "expecting batch dimension of each tensor to be 1"
    # remove batch dimension for each tensor
    input_ids = [x.squeeze(0) for x in input_ids]
    return collator({"input_ids": input_ids})["input_ids"]


def generate(
        examples,
        data_collator=None,
        generate_without_watermark=None,
        generate_with_watermark=None,
        watermark_processor=None,
        tokenizer=None,
        args=None,
):
    input_ids = collate_batch(input_ids=examples["input_ids"], collator=data_collator).to(
        args.device)
    with torch.no_grad():
        if args.generation_seed is not None:
            torch.manual_seed(args.generation_seed)
        output_without_watermark = generate_without_watermark(input_ids=input_ids)

        if args.generation_seed is not None:
            torch.manual_seed(args.generation_seed)
        output_with_watermark = generate_with_watermark(input_ids=input_ids)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:, input_ids.shape[-1]:]
        output_with_watermark = output_with_watermark[:, input_ids.shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(
        output_without_watermark, skip_special_tokens=True
    )
    decoded_output_with_watermark = tokenizer.batch_decode(
        output_with_watermark, skip_special_tokens=True
    )
    examples.update(
        {
            "no_wm_output": decoded_output_without_watermark,
            "w_wm_output": decoded_output_with_watermark,
            "no_wm_output_length": (output_without_watermark != tokenizer.pad_token_id).sum(
                dim=-1).tolist(),
            "w_wm_output_length": (output_with_watermark != tokenizer.pad_token_id).sum(
                dim=-1).tolist(),
        }

    )

    examples.pop("input_ids")

    if watermark_processor.spike_entropies is not None:
        examples["spike_entropies"] = watermark_processor._get_and_clear_stored_spike_ents()
        examples["spike_entropies"] = [
            ents[:num_toks]
            for ents, num_toks in zip(examples["spike_entropies"], examples["w_wm_output_length"])
        ]

    return examples
