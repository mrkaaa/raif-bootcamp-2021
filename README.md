# Решение команды RaifGPT-3

Будет тут :)

## Docker

Сборка:
```bash
./bin/build.sh
```

Запуск:
```bash
./bin/run.sh
```

## Описание API

Сначала вызывается `/predict`, затем в `/result_question` — приходит результат предыдущего ответа

### /predict

 В `/predict` приходит:
```json
{   
    'number of game': 5,
    'question': "Что есть у Пескова?",
    'answer_1': "Усы",
    'answer_2': "Борода",
    'answer_3': "Лысина",
    'answer_4': "Третья нога",
    'question money': 4000,
    'saved money': 1000,
    'available help': ["fifty fifty", "can mistake", "new question"]
}
```

Поддерживаемые ответы:
```json
{
    'answer': 1
}
{   
    'help': "fifty fifty",
}
{   
    'help': "can mistake",
    'answer': 1,
}
{   
    'end game': "take money",
}
```

### /result_question

В `/result_question` приходит:

```json
{
    'number of game': 5,
    'question': "Что есть у Пескова?",
    'answer': 1,
    'bank': 4000,
    'saved money': 1000,
    'response type': "good"
}
```