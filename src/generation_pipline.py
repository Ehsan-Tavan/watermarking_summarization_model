# =========================== Third Party libs ========================
import os
import tqdm
from functools import partial
from transformers import LogitsProcessorList, DataCollatorWithPadding
from torch.utils.data import DataLoader
# ============================ My packages ============================
from src.configurations import BaseConfig
from src.data_reader import read_csv, write_json
from src.utils import load_model, WatermarkLogitsProcessor, generate, prepare_example, \
    load_detector, compute_z_scores
from src.dataset import Dataset

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    for d in [0.5, 1, 2, 5, 10]:
        ARGS.delta = d
        ARGS.evaluation_output_file = f"cnn_dailymail_test_delta={d}_gamma={ARGS.gamma}.json"

        DATA = read_csv(os.path.join(ARGS.data_dir, ARGS.test_data_file))
        DATA = DATA.sample(n=1000)

        DATA = prepare_example(DATA)

        MODEL, TOKENIZER = load_model(path=ARGS.lm_path, device=ARGS.device)
        GEN_KWARGS = {"num_beams": 1,
                      "max_length": 80,
                      "min_length": 20}

        if ARGS.use_sampling:
            GEN_KWARGS.update(
                dict(
                    do_sample=True,
                    top_k=ARGS.top_k,
                    top_p=ARGS.top_p,
                    temperature=ARGS.sampling_temp,
                    num_beams=ARGS.num_beams,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )
            )

        watermark_processor = WatermarkLogitsProcessor(
            vocab=list(TOKENIZER.get_vocab().values()),
            gamma=ARGS.gamma,
            delta=ARGS.delta,
            seeding_scheme=ARGS.seeding_scheme,
            store_spike_ents=ARGS.store_spike_ents,
            select_green_tokens=True,
        )

        GENERATE_WITHOUT_WATERMARK = partial(MODEL.generate, **GEN_KWARGS)
        GENERATE_WITH_WATERMARK = partial(
            MODEL.generate, logits_processor=LogitsProcessorList([watermark_processor]),
            **GEN_KWARGS
        )

        # construct the collator
        DATA_COLLATOR = DataCollatorWithPadding(tokenizer=TOKENIZER, padding=True,
                                                pad_to_multiple_of=8)

        generation_partial = partial(
            generate,
            data_collator=DATA_COLLATOR,
            generate_without_watermark=GENERATE_WITHOUT_WATERMARK,
            generate_with_watermark=GENERATE_WITH_WATERMARK,
            watermark_processor=watermark_processor,
            tokenizer=TOKENIZER,
            args=ARGS,
        )

        DATASET = Dataset(data=DATA, tokenizer=TOKENIZER, max_len=512, device=ARGS.device)
        DATA_LOADER = DataLoader(DATASET, batch_size=1, shuffle=False)

        WATERMARK_DETECTOR = load_detector(tokenizer=TOKENIZER, args=ARGS)

        COMPUTE_Z_SCORES_PARTIAL = partial(
            compute_z_scores,
            watermark_detector=WATERMARK_DETECTOR,
            args=ARGS
        )

        OUTPUTS = []
        for batch in tqdm.tqdm(DATA_LOADER):
            OUTPUTS.append(COMPUTE_Z_SCORES_PARTIAL(generation_partial(batch)))

        write_json(OUTPUTS, os.path.join(ARGS.assets_dir, ARGS.evaluation_output_file))
