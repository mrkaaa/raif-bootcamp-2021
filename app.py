import logging
import random
import typing

from flask import Flask, request

from models import GPTModel, RuBertModel


logging.basicConfig(level=logging.INFO)

APP: Flask = Flask(__name__)
MODEL: GPTModel = GPTModel("sberbank-ai/rugpt3large_based_on_gpt2")
LOGGER: logging.Logger = logging.getLogger("RaifGPT-3")


@APP.route("/predict", methods=["POST"])
def predict():
    data: dict = request.form
    available_help = set(data["available help"])
    question: str = data["question"]
    variants: typing.List[str] = [data[x] for x in ["answer_1", "answer_2", "answer_3", "answer_4"] if data[x]]

    prediction, score = MODEL.predict(variants, question)

    if score > 28 and data['question money'] != 2000:
        if "new question" in available_help:
            return {"help": "new question"}
        if "can mistake" in available_help:
            return {"answer": prediction, "help": "can mistake"}
        if "fifty fifty" in available_help:
            return {"help": "fifty fifty"}
        return {"end game": "take money"}

    return {"answer": prediction}


@APP.route("/result_question", methods=["POST"])
def result_question():
    # data = request.form.get("data")
    return {"data": "ok"}


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=12304, debug=True)
