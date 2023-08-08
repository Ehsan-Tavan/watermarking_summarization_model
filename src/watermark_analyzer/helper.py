# ============================ Third Party libs ============================
from typing import List
import evaluate


def calculate_rouge_score(predictions: List[str], references: List[List]):
    evaluator = evaluate.load("analyzer/metrics/rouge/")
    rouge_type2value = evaluator.compute(predictions=predictions, references=references,
                                         tokenizer=lambda x: x.split())
    return rouge_type2value


def convert_list_of_dictionaries_to_one_dictionary(dictionaries: List[dict]):
    merged_dict = {}
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            if isinstance(value, list):
                if key not in merged_dict:
                    merged_dict[key] = []
                merged_dict[key].extend(value)
            else:
                merged_dict.setdefault(key, []).append(value)
    return merged_dict


def calculate_average(data: list):
    total_sum = sum(data)
    average = total_sum / len(data)
    return average
