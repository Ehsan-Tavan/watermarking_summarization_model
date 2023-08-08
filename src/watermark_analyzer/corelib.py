# =========================== Third Party libs ========================
from typing import List
import mauve
from simcse import SimCSE
# ============================ My packages ============================
from .helper import calculate_rouge_score, convert_list_of_dictionaries_to_one_dictionary, \
    calculate_average
from .metrics.coherence import Coherence


class CoreLib:
    def __init__(self, data: List[dict], simcse_model_path: str, device: str = "cpu"):
        self.data = data

        self.data = convert_list_of_dictionaries_to_one_dictionary(self.data)

        self.model = None
        self.load_model(model_path=simcse_model_path, device=device)

        self.coherence = Coherence(model=self.model)

    def load_model(self, model_path: str, device: str = "cpu"):
        self.model = SimCSE(model_path, device=device)

    def get_embedding(self, texts: List[str]):
        return self.model.encode(texts)

    def run(self):
        metric2value = {}
        no_wm_rouge_score = calculate_rouge_score(predictions=self.data["no_wm_output"],
                                                  references=self.data["summary"])
        w_wm_rouge_score = calculate_rouge_score(predictions=self.data["w_wm_output"],
                                                 references=self.data["summary"])

        metric2value["no_wm_rouge1"] = no_wm_rouge_score["rouge1"]
        metric2value["no_wm_rouge2"] = no_wm_rouge_score["rouge2"]
        metric2value["no_wm_rougeL"] = no_wm_rouge_score["rougeL"]

        metric2value["w_wm_rouge1"] = w_wm_rouge_score["rouge1"]
        metric2value["w_wm_rouge2"] = w_wm_rouge_score["rouge2"]
        metric2value["w_wm_rougeL"] = w_wm_rouge_score["rougeL"]

        no_wm_z_score = calculate_average(self.data["no_wm_output_z_score"])
        w_wm_z_score = calculate_average(self.data["w_wm_output_z_score"])

        metric2value["no_wm_z_score"] = no_wm_z_score
        metric2value["w_wm_z_score"] = w_wm_z_score

        no_wm_coherence = self.coherence.calculate_coherence(references=self.data["summary"],
                                                             predictions=self.data["no_wm_output"])
        w_wm_coherence = self.coherence.calculate_coherence(references=self.data["summary"],
                                                            predictions=self.data["w_wm_output"])

        metric2value["no_wm_coherence"] = no_wm_coherence
        metric2value["w_wm_coherence"] = w_wm_coherence

        no_wm_mauve = mauve.compute_mauve(p_features=self.get_embedding(self.data["summary"]),
                                          q_features=self.get_embedding(self.data["no_wm_output"]))
        w_wm_mauve = mauve.compute_mauve(p_features=self.get_embedding(self.data["summary"]),
                                         q_features=self.get_embedding(self.data["w_wm_output"]))

        metric2value["no_wm_mauve"] = no_wm_mauve.mauve
        metric2value["w_wm_mauve"] = w_wm_mauve.mauve

        return metric2value
