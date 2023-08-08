import numpy as np
from simcse import SimCSE


class Coherence:
    def __init__(self, model):
        self.model = model

    def calculate_coherence(self, references, predictions):
        similarities = self.model.similarity(references, predictions)
        similarities = np.array(similarities)
        coherence_score = similarities.trace() / len(similarities)
        return coherence_score
