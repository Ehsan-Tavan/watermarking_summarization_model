# ============================ Third Party libs ============================
import argparse
from pathlib import Path


class BaseConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--device",
                                 type=str,
                                 default="cuda:0")

    def generation(self) -> None:
        self.parser.add_argument("--use_sampling",
                                 type=bool,
                                 default=False,
                                 help="Whether to perform sampling during generation. "
                                      "(non-greedy decoding)")
        self.parser.add_argument("--max_new_tokens",
                                 type=int,
                                 default=200,
                                 help="The number of tokens to generate using the model, "
                                      "and the num tokens removed from real text sample")
        self.parser.add_argument("--num_beams",
                                 type=int,
                                 default=10,
                                 help="The number of beams to use where '1' is no beam search.")
        self.parser.add_argument("--top_k",
                                 type=int,
                                 default=50,
                                 help="The top k to use when generating using top_k version of "
                                      "multinom sampling")
        self.parser.add_argument("--top_p",
                                 type=float,
                                 default=1.0,
                                 help="The top p to use when generating using top_p "
                                      "version of sampling")

        self.parser.add_argument("--sampling_temp",
                                 type=float,
                                 default=0.7,
                                 help="The temperature to use when generating using "
                                      "multinom sampling")
        self.parser.add_argument("--gamma",
                                 type=float,
                                 default=0.25,
                                 help="The ratio of tokens to put in the greenlist when splitting "
                                      "the vocabulary",
                                 )
        self.parser.add_argument("--delta",
                                 type=float,
                                 default=5.0,
                                 help="The amount of bias (absolute) to add to the logits in the "
                                      "whitelist half of the vocabulary at every step",
                                 )
        self.parser.add_argument("--seeding_scheme",
                                 type=str,
                                 default="simple_1",
                                 help="The seeding procedure to use for the watermark.")
        self.parser.add_argument("--generation_seed",
                                 type=int,
                                 default=123,
                                 help="Seed for setting the torch rng prior to generation using "
                                      "any decoding scheme with randomness.")
        self.parser.add_argument("--is_decoder_only_model",
                                 type=bool,
                                 default=False)

        self.parser.add_argument("--store_spike_ents",
                                 type=bool,
                                 default=True,
                                 help="Whether to store the spike entropies while generating with "
                                      "watermark processor. ")

    def evaluation(self):
        self.parser.add_argument("--return_green_token_mask",
                                 type=bool,
                                 default=True,
                                 help="Whether to return the mask marking which tokens are green "
                                      "from the watermark detector.")
        self.parser.add_argument("--verbose",
                                 type=bool,
                                 default=True)
        self.parser.add_argument("--compute_scores_at_T",
                                 type=bool,
                                 default=True,
                                 help="Whether to compute (applicable) metrics at each T index in "
                                      "the output/text columns.")

        self.parser.add_argument("--detection_z_threshold",
                                 type=float,
                                 default=4.0,
                                 help="The test statistic threshold for the detection hypothesis "
                                      "test.",
                                 )

    def add_path(self) -> None:
        self.parser.add_argument("--assets_dir",
                                 type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/")

        self.parser.add_argument("--evaluation_output_file",
                                 type=str,
                                 default="cnn_dailymail_test.json")

        self.parser.add_argument("--data_dir",
                                 type=str,
                                 default=Path(__file__).parents[
                                             2].__str__() + "/data/cnn_dailymail")

        self.parser.add_argument("--test_data_file",
                                 type=str,
                                 default="test.csv")

        self.parser.add_argument("--lm_path",
                                 type=str,
                                 default="/mnt/disk2/Language_models/t5_large_finetuned_xsum_cnn")

    def get_config(self) -> argparse.Namespace:
        self.add_path()
        self.generation()
        self.evaluation()
        return self.parser.parse_args()
