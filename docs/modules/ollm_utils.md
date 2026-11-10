# 💬 ollm_utils.py — Ollama и локальные LLM

Чат с моделями Ollama, классификация моделей по имени и автоматическая
настройка Jupyter AI и OpenCode на локальный сервер.

**Зависимости:** `requests`, `beautifulsoup4`, `utils.py`; опционально
`jupyter-ai` и бинарник `opencode`.

## 💬 Поговорить с моделью

```python
import os

os.environ['OLLAMA_HOST'] = 'http://localhost:11434'

from ollm_utils import Chat

chat = Chat(model='qwen2.5-coder:7b', temperature=0.0)
answer = chat('Объясни, что делает функция mpmap.')
print(answer)
```

`Chat` хранит историю диалога в `messages`, поэтому следующий вызов помнит
предыдущие реплики. `chat.reset()` очищает историю.

## 📤 Передать файлы

```python
answer = chat(
    'Найди ошибку в этом модуле.',
    file=['src/train.py', 'src/model.py'],
)
```

Содержимое файлов добавляется к сообщению с заголовками. Для одного файла
можно передать строку или `Path`.

## 🏷️ Классифицировать модели

```python
from ollm_utils import hosts2chat_embd_cmpl_models

chat_models, embd_models, cmpl_models = hosts2chat_embd_cmpl_models(
    ['http://localhost:11434'],
)

print(list(chat_models))
print(list(embd_models))
print(list(cmpl_models))
```

Функция опрашивает `/api/tags` и делит модели на диалоговые, эмбеддинговые и
кодогенерирующие. Тип определяется по тегам страницы модели на `ollama.com`,
затем по эвристикам имени (`embed`, `code`, `chat`).

## ⚙️ Настроить Jupyter AI

```python
from ollm_utils import set_jupyter_ai_settings

path = set_jupyter_ai_settings()
print(path)
```

Функция определяет версию `jupyter-ai` и пишет конфиг нужного формата: v2
или v3. Если `jupyter-ai` не установлен, возвращается пустая строка. Адрес
сервера берётся из `OLLAMA_HOST` или из аргумента `hosts`.

## ⚙️ Настроить OpenCode

```python
from ollm_utils import set_opencode_settings

path = set_opencode_settings()
print(path)
```

Конфиг `~/.config/opencode/opencode.jsonc` дополняется провайдерами Ollama.
Функция отключает генерацию заголовков на внешних серверах и sharing сессий.

## 🚀 Запустить настройку целиком

```bash
python ollm_utils.py
```

При запуске как скрипта вызываются обе настройки. Это удобно в
`docker/cmd.sh`, где `OLLAMA_HOST` уже передан в контейнер.

## 🔗 Нормализовать адрес

```python
from ollm_utils import normalize_host

print(normalize_host('localhost'))  # http://localhost:11434
```

Функция добавляет схему и порт по умолчанию. Пустая строка означает, что
сервер не задан.

## ⚠️ Ограничения

- `model_name2type` ходит на `ollama.com`; без сети тип определяется только по
  эвристикам имени и может быть `unknown`.
- `set_jupyter_ai_settings` работает только при установленном `jupyter-ai` и
  перезаписывает поля провайдеров в его конфиге.
- `set_opencode_settings` меняет `agent.title` и `share` в конфиге OpenCode;
  остальные настройки сохраняются.
- `Chat` не поддерживает потоковый ответ и не ограничивает длину истории:
  долгий диалог может упереться в контекст модели.
- При `temperature=0` каждый вызов делает дополнительный прогрев модели,
  поэтому первый ответ приходит медленнее.
