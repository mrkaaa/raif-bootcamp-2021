import logging
import random
import typing

from flask import Flask, request

from models import RuBertModel


logging.basicConfig(level=logging.INFO)

APP: Flask = Flask(__name__)
MODEL: RuBertModel = RuBertModel()
LOGGER: logging.Logger = logging.getLogger("RaifGPT-3")


@APP.route("/predict", methods=["POST"])
def predict():
    data: dict = request.form.get("data")

    question: str = data["question"]
    variants: typing.List[str] = [data["answer_1"], data["answer_2"], data["answer_3"], data["answer_4"]]

    return random.choice([{"answer": MODEL.predict(variants, question)}, {"end game": "take money"}])


@APP.route("/result_question", methods=["POST"])
def result_question():
    # data = request.form.get("data")
    return {"data": "ok"}


if __name__ == "__main__":
    APP.run(host="127.0.0.1", port=12304)
