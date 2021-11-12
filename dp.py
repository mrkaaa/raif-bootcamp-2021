import logging
import time

from colorama import Fore

from deeppavlov import build_model, configs


logging.basicConfig(level=logging.ERROR)


def timed_inference(model, question: str, context: str, print_q_v: bool = True):
    start_time = time.perf_counter()
    prediction = model([context], [question])[0][0]
    predict_time = time.perf_counter() - start_time

    if print_q_v:
        print(Fore.BLUE + f"Question: {question}")
        print(Fore.YELLOW + f"Variants: {context}")
    print(Fore.GREEN + f"[{predict_time:.3f}] Predicted: `{prediction}`")


model = build_model(configs.squad.squad)

timed_inference(model, "Раз два три четыре пять вышел зайчик что делать?", "Поплавать. Погулять. Покурить. Попрыгать.")

question: str = ""
try:
    while question != "q":
        question = input(Fore.BLUE + "Вопрос: ")
        variants: str = input(Fore.YELLOW + "Варианты: ")

        timed_inference(model, question, variants, print_q_v=False)
except KeyboardInterrupt:
    print(Fore.RED + "Abort")
