FROM python:3.7.9

COPY . .

RUN pip install poetry && poetry config virtualenvs.create false && poetry install
RUN python -c "from models import GPTModel, RLModel; GPTModel('sberbank-ai/rugpt3large_based_on_gpt2'); RLModel()"

CMD ["python", "app.py"]