import difflib
import logging
import random
import typing

from flask import Flask, request

from deeppavlov import build_model, configs


logging.basicConfig(level=logging.INFO)

APP: Flask = Flask(__name__)
LOGGER: logging.Logger = logging.getLogger("RaifGPT-3")
MODEL = build_model(configs.squad.squad)


def find_match(variants: typing.List[str], prediction: str) -> int:
    best_score: float = -1.0
    best_idx: int = -1

    for i, variant in enumerate(variants):
        score: float = difflib.SequenceMatcher(a=variant, b=prediction).ratio()

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


@APP.route("/predict", methods=["POST"])
def predict():
    data: dict = request.form.get("data")

    question: str = data["question"]
    variants: typing.List[str] = [data["answer_1"], data["answer_2"], data["answer_3"], data["answer_4"]]

    concatenated_variants: str = ";".join(variants)

    prediction_details: list = MODEL([concatenated_variants], [question])
    prediction: str = prediction_details[0][0]
    match: int = find_match(variants=variants, prediction=prediction)

    LOGGER.info("Question: {question}; Variants: {variants}; Answer: {variants[match]}")

    return random.choice([{"answer": match}, {"end game": "take money"}])


@APP.route("/result_question", methods=["POST"])
def result_question():
    # data = request.form.get("data")
    return {"data": "ok"}


if __name__ == "__main__":
    APP.run(host="127.0.0.1", port=12304)
