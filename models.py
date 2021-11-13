import difflib
import typing

from deeppavlov import build_model, configs
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from rl.agent import SimpleNNAgent
from rl.rl_env import str_to_embbeding
from model_data import DATA_PATH


class RuBertModel:
    def __init__(self):
        self.model = build_model(configs.squad.squad)

    def predict(self, variants: typing.List[str], question: str) -> int:
        concatenated_variants: str = "; ".join(variants)
        prediction_details: list = self.model([concatenated_variants], [question])
        prediction: str = prediction_details[0][0]

        return self.find_match(variants=variants, prediction=prediction)

    @staticmethod
    def find_match(variants: typing.List[str], prediction: str) -> int:
        best_score: float = -1.0
        best_idx: int = -1

        for i, variant in enumerate(variants, start=1):
            score: float = difflib.SequenceMatcher(a=variant, b=prediction).ratio()

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx


class RLModel:
    def __init__(self):
        self.model = SimpleNNAgent((5, 768), 5)
        self.model.load_state_dict(DATA_PATH)
        self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
        self.bertmodel = BertModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

    def predict(self, variants: typing.List[str], question: str) -> int:
        state = self.get_state(variants, question)
        prediction = self.model.sample_greedy(state)

        if prediction == 4.:
            return {"end game": "take money"}
        return {"answer": prediction +1}

    def get_state(self, variants, question):
        sample = pd.DataFrame(['x'], columns=['question'])

        sample['question'] = question

        for index, variant in enumerate(variants):
            sample[f'variant_{index}'] = variant

        cols = sample.columns.values
        for column in cols:
            sample[f'emb_{column}'] = str_to_embbeding(sample[f'{column}'],
                                                       self.tokenizer,
                                                       self.bertmodel)
            sample = sample.drop(columns=[column])
        sample['state'] = sample.to_numpy().tolist()

        return sample.state.apply(lambda x: np.array(x)).iloc[0]


