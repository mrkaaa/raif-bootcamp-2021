import difflib
import typing

from deeppavlov import build_model, configs


class RuBertModel:
    def __init__(self):
        self.model = build_model(configs.squad.squad)

    def predict(self, variants: typing.List[str], question: str) -> dict:
        concatenated_variants: str = "; ".join(variants)
        prediction_details: list = self.model([concatenated_variants], [question])
        prediction: str = prediction_details[0][0]

        return {"answer": self.find_match(variants=variants, prediction=prediction)}

    @staticmethod
    def find_match(variants: typing.List[str], prediction: str) -> int:
        best_score: float = -1.0
        best_idx: int = -1

        for i, variant in enumerate(variants):
            score: float = difflib.SequenceMatcher(a=variant, b=prediction).ratio()

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx
