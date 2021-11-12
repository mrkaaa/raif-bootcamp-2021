FROM python:3.7.9

COPY . .

RUN pip install poetry && poetry config virtualenvs.create false && poetry install
RUN python -c "from deeppavlov import build_model, configs; build_model(configs.squad.squad, download=True)"

CMD ["python", "app.py"]