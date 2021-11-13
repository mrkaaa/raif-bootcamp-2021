import difflib
import typing

from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from rl.agent import SimpleNNAgent
from rl.rl_env import str_to_embbeding
from model_data import DATA_PATH
import torch


class RuBertModel:
    def __init__(self):
        from deeppavlov import build_model, configs

        self.model = build_model(configs.squad.squad)

    def predict(self, variants: typing.List[str], question: str) -> typing.Tuple[int, float]:
        concatenated_variants: str = "; ".join(variants)
        prediction_details: list = self.model([concatenated_variants], [question])
        prediction: str = prediction_details[0][0]

        return self.find_match(variants=variants, prediction=prediction)

    @staticmethod
    def find_match(variants: typing.List[str], prediction: str) -> typing.Tuple[int, float]:
        best_score: float = -1.0
        best_idx: int = -1

        for i, variant in enumerate(variants, start=1):
            score: float = difflib.SequenceMatcher(a=variant, b=prediction).ratio()

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx, best_score


class GPTModel:
    def __init__(self, model_name: str):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.model_name: str = model_name
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def calc_ppl(self, input_ids, stride=128) -> float:
        max_length = self.model.config.n_positions

        nlls = []
        for i in range(0, input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        return float(torch.exp(torch.stack(nlls).sum() / end_loc))

    def predict(self, variants: typing.List[str], question: str) -> typing.Tuple[int, float]:
        best_score: float = float("inf")
        best_id: int = -1
        for i, answer in enumerate(variants, start=1):
            text: str = f"Вопрос: {question}. Ответ: {answer}."
            score = self.calc_ppl(self.tokenizer.encode(text, return_tensors="pt"))

            if score < best_score:
                best_score = score
                best_id = i

        return best_id, best_score


class RLModel:
    def __init__(self):
        self.model = SimpleNNAgent((5, 768), 5)
        self.model.load_state_dict(torch.load(f'{DATA_PATH}/torch_model'))
        self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
        self.bertmodel = BertModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

    def predict(self, variants: typing.List[str], question: str) -> int:
        state = self.get_state(variants, question)
        prediction = self.model.sample_greedy(state)

        if prediction == 4.:
            return {"end game": "take money"}
        return {"answer": prediction + 1}

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


