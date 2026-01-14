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
| `file2context()`                | преобразование файлов в текстовое представление для контекста      |
<!-- markdownlint-enable MD013 -->

### Классы

| Класс  | Описание                                    |
|--------|---------------------------------------------|
| `Chat` | объект для ведения диалога с моделью Ollama |

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

#### Параметры set_jupyter_ai_settings

- `hosts` (Hosts) — адреса Ollama-серверов. Аналогично `hosts2chat_embd_cmpl_models`.

#### Возвращаемое значение set_jupyter_ai_settings

Возвращает путь до конфигурационного файла Jupyter AI.

#### Особенности set_jupyter_ai_settings

- Требует установленного `jupyter_ai`. Если не установлен, возвращает пустую строку.
- Читает существующий конфигурационный файл или создаёт новый.
- Заполняет конфигурацию моделями, полученными от `hosts2chat_embd_cmpl_models`.
- Устанавливает настройку `send_with_shift_enter` в `True`.

### Функция `file2context`

Преобразует файл или список файлов в текстовое представление с форматированными
заголовками.

```python
def file2context(
    file: list[str | Path] | str | Path
) -> str
```

#### Параметры file2context

- `file` — путь к файлу, список путей или объект `Path`.

#### Возвращаемое значение file2context

Строка, состоящая из содержимого всех файлов, разделённых заголовками с именами файлов и
путями.

---

## Описание классов

### Класс `Chat`

Основной интерфейс для взаимодействия с моделями Ollama через чат.

```python
class Chat:
    def __init__(
        self,
        host: str = "",
        model: str = "",
        temperature: float = 0.0,
        timeout: int = 300,
        seed: int = 42,
    ) -> None
```

#### Параметры конструктора

- `host` — адрес Ollama-сервера (по умолчанию берётся из `OLLAMA_HOST`)
- `model` — имя модели (если не указана, выбирается первая доступная)
- `temperature` — температура инференса
- `timeout` — время ожидания ответа в секундах
- `seed` — seed для генерации (в сочетании с `temperature=0` даёт негарантированную
   детерминированность)

#### Основные методы

##### `__call__` — отправка сообщения модели и получение ответа

```python
def __call__(
    self,
    message: str,
    file: list[str | Path] | str | Path | None = None,
) -> str
```

##### `get_models` — получение доступных моделей

```python
def get_models(self) -> tuple[list[str], list[str], list[str]]
```

##### `reset` — очистка истории диалога, позволяя начать новый диалог

```python
def reset(self) -> None
```

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

### Работа с файлами

```python
# Преобразование одного файла
context = file2context("document.txt")

# Преобразование списка файлов
context = file2context(["chapter1.txt", "chapter2.txt", "chapter3.txt"])
```

### Использование класса Chat

```python
# Создание экземпляра чата
chat = Chat(host="http://localhost:11434", model="llama2:7b")

# Простой диалог
response = chat("Привет!")
print(response)

# Диалог с прикреплённым файлом
response = chat("Объясни этот код", file="example.py")

# Диалог с несколькими файлами
response = chat("Сравни эти документы", file=["doc1.txt", "doc2.txt"])

# Очистка истории диалога
chat.reset()

# Получение списка доступных моделей
chat_models, embd_models, cmpl_models = chat.get_models()
print(f"Доступные чат-модели: {chat_models}")
```

### Расширенное использование Chat

```python
# Создание чата с параметрами
chat = Chat(
    host="http://localhost:11434",
    model="codellama:7b",
    temperature=0.7,  # Более креативные ответы
    timeout=600,      # Увеличенное время ожидания
    seed=12345,       # Конкретный seed
)

# Многошаговый диалог
response1 = chat("Напиши функцию сложения на Python")
response2 = chat("А теперь добавь проверку типов")
response3 = chat("Сделай то же самое на JavaScript", file="python_code.py")
```

---

## Примечания

- Для полноценной работы функций, связанных с парсингом сайта Ollama, требуется
подключение к интернету.
- Настройка Jupyter AI требует установки соответствующего пакета.
- Класс `Chat` предоставляет основной интерфейс для взаимодействия с моделями Ollama,
  поддерживая историю диалога, работу с файлами и настраиваемые параметры запросов.
- Функция `file2context` позволяет включать содержимое файлов в контекст запроса к модели,
  что полезно для анализа кода, документов и других текстовых данных.
- При `temperature=0.0` и указании `seed` обеспечивается детерминированность ответов
  модели, что полезно для тестирования и воспроизводимости.
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
