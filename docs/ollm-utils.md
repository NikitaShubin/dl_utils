# Модуль [`ollm_utils.py`](../ollm_utils.py "Перейти к модулю")

Предоставляет функциональность для работы с Ollama-сервисом. Основное назначение —
автоматическое определение типа моделей (чат, эмбеддинги, кодогенерация) и настройка
Jupyter AI для работы с моделями Ollama.

---

## Основные компоненты

### Функции

<!-- markdownlint-disable MD013 -->
| Функция                         | Описание                                                           |
|---------------------------------|--------------------------------------------------------------------|
| `model_name2type()`             | определение типа модели по её имени                                |
| `hosts2chat_embd_cmpl_models()` | разделение моделей на три категории: chat, embeddings, completions |
| `set_jupyter_ai_settings()`     | настройка Jupyter AI для работы с моделями Ollama                  |
| `url2tags()`                    | извлечение тегов со страницы модели на сайте Ollama                |
| `host2models_info()`            | получение списка моделей с Ollama-сервера                          |
<!-- markdownlint-enable MD013 -->

---

## Описание функций

### Функция `model_name2type`

Определяет тип модели по её имени, используя парсинг страницы модели на сайте Ollama.

```python
def model_name2type(model_name: str) -> str
```

#### Алгоритм работы

1. **Очистка имени модели** от версии (например, удаление `:7b`, `:latest`, `:q4_K_M`).
1. **Парсинг страницы модели** на сайте Ollama
(`https://ollama.com/library/{model_name}`).
1. **Извлечение тегов модели** со страницы.
1. **Классификация** на основе тегов и ключевых слов в имени модели.

### Функция `hosts2chat_embd_cmpl_models`

Составляет словари моделей, разделённых по типам, для использования в конфигурационном
файле Jupyter AI.

```python
def hosts2chat_embd_cmpl_models(
    hosts: Hosts = None
) -> tuple[Fields, Fields, Fields]
```

### Функция `set_jupyter_ai_settings`

Настраивает Jupyter AI для работы с моделями Ollama. Создаёт или обновляет
конфигурационный файл Jupyter AI.

```python
def set_jupyter_ai_settings(
    hosts: Hosts = None
) -> str
```

#### Параметры

- `hosts` (Hosts) — адреса Ollama-серверов. Аналогично `hosts2chat_embd_cmpl_models`.

#### Возвращаемое значение

Возвращает путь до конфигурационного файла Jupyter AI.

#### Особенности

- Требует установленного `jupyter_ai`. Если не установлен, возвращает пустую строку.
- Читает существующий конфигурационный файл или создаёт новый.
- Заполняет конфигурацию моделями, полученными от `hosts2chat_embd_cmpl_models`.
- Устанавливает настройку `send_with_shift_enter` в `True`.

---

## Примеры использования

### Определение типа модели

```python
model_type = model_name2type("llama2:7b")
print(model_type)  # 'chat'
```

### Получение моделей с сервера

```python
chat_models, embd_models, cmpl_models = hosts2chat_embd_cmpl_models(
    ["http://localhost:11434"]
)
```

### Настройка Jupyter AI

```python
config_path = set_jupyter_ai_settings(['<host1>:<port1>', ... '<hostn>:<portn>'])
print(f"Конфигурация сохранена в: {config_path}")
```

---

## Примечания

- Для полноценной работы функций, связанных с парсингом сайта Ollama, требуется
подключение к интернету.
- Настройка Jupyter AI требует установки соответствующего пакета.
- При прямом запуске скрипта:

  ```bash
  ./ollm_utils.py
  ```

  или через интерпретатор:

  ```bash
  python ollm_utils.py
  ```

  выполняется функция `set_jupyter_ai_settings()` без параметров, что означает
использование адреса сервера из переменной окружения `OLLAMA_HOST` для настройки
Jupyter AI.
