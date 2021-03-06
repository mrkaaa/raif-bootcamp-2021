import logging
import random
import typing

from flask import Flask, request

from models import GPTModel, RLModel, RuBertModel


logging.basicConfig(level=logging.INFO)

APP: Flask = Flask(__name__)
MODEL: GPTModel = GPTModel("sberbank-ai/rugpt3large_based_on_gpt2")
RL_MODEL: RLModel = RLModel()
LOGGER: logging.Logger = logging.getLogger("RaifGPT-3")


@APP.route("/predict", methods=["POST"])
def predict():
    data: dict = request.form.to_dict(flat=False)
    available_help = set(data["available help"])
    question: str = data["question"][0]
    all_variants = [data[x][0] for x in ["answer_1", "answer_2", "answer_3", "answer_4"]]
    variants: typing.List[str] = [
        data[x][0] if data[x][0] and data[x][0].lower() != "неверный ответ" else "_"
        for x in ["answer_1", "answer_2", "answer_3", "answer_4"]
    ]

    prediction, score = MODEL.predict(variants, question)

    if score > 35 and data["question money"] not in (100, 2000):
        if "new question" in available_help:
            return {"help": "new question"}
        if "fifty fifty" in available_help:
            return {"help": "fifty fifty"}
        if "can mistake" in available_help:
            return {"answer": prediction, "help": "can mistake"}
        if len(variants) == 1:
            return {"answer": [i for i, v in enumerate(all_variants, 1) if v is not None][0]}
        return {"end game": "take money"}

    return {"answer": prediction}


@APP.route("/predict2", methods=["POST"])
def predict2():
    data: dict = request.form
    available_help = set(data["available help"])
    question: str = data["question"]
    variants: typing.List[str] = [data[x] for x in ["answer_1", "answer_2", "answer_3", "answer_4"]]

    prediction = RL_MODEL.predict(variants, question)
    return prediction


@APP.route("/result_question", methods=["POST"])
def result_question():
    # data = request.form.get("data")
    return {"data": "ok"}


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=12304)
