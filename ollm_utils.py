#!/usr/bin/env python3
"""ollm_utils.py.

********************************************
*        Работа с Ollama-сервисом.         *
*                                          *
*   Модуль для взаимодействия с моделями   *
*   Ollama, их классификации и настройки   *
*       интеграции с Jupyter AI.           *
*                                          *
* Основные возможности:                    *
* • автоматическое определение типа модели *
*   (чат,эмбеддинги, кодогенерация) по     *
*   имени;                                 *
* • парсинг информации о моделях с         *
*   официального сайта Ollama;             *
* • получение списка доступных моделей с   *
*   Ollama-серверов;                       *
* • автоматическая настройка Jupyter AI    *
*   для работы с моделями Ollama.          *
*                                          *
* Основные функции:                        *
* • hosts2chat_embd_cmpl_models -          *
*   классификация доступных на сервере     *
*   моделей;                               *
* • set_jupyter_ai_settings() - настройка  *
*   Jupyter AI.                            *
*                                          *
* Вызов модуля как исполняемого файла      *
* запускает set_jupyter_ai_settings без    *
* параметров - доступный Ollama-сервер     *
* читается из переменной окружения         *
* OLLAMA_HOST и jupyter-ai автоматически   *
* настраивается на использование её        *
* моделей.                                 *
*                                          *
********************************************
.
"""

import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    import jupyter_ai  # type: ignore[import-untyped]

    JUPYTER_AI_AVAILABLE = True
except ImportError:
    JUPYTER_AI_AVAILABLE = False

from utils import json2obj, mkdirs, obj2json

# Имя переменной окружения, хранящей адрес Ollama-сервера:
env_var_host: str = 'OLLAMA_HOST'


# Типы:
Fields = dict[str, dict[str, str]]  # Описание моделей
Hosts = set[str] | list[str] | tuple[str, ...] | None  # Множество серверов


# Варианты селекторов для тегов (зависит от структуры страницы):
possible_selectors = [
    'div.tags a',
    '.tag',
    '.model-tag',
    '[class*="tag"]',
    'span.badge',
    '.model-tags span',
]

# Множества ключевых слов в тексте для каждого тега:
tag2possible_keywords = {
    'embeddings': {
        'embed',
        'embedding',
    },
    'completions': {
        'completion',
    },
    'chat': {
        'chat',
    },
}
# Используются только если теги не найдены в явном виде.

# Множества ключевых слов в тегах для каждого типа модели:
type2possible_tags = {
    'embeddings': {'embeddings', 'embed', 'embedding', 'vector'},
    'completions': {
        'code',
        'coder',
        'coding',
        'programming',
        'codegen',
        'completion',
    },
    'chat': {
        'chat',
        'conversation',
        'dialogue',
        'assistant',
        'thinking',
    },
}
# В порядке приоритетов: embed > code > chat.


def url2tags(url: str) -> list[str]:
    """Извлекает теги из страницы модели."""
    tags = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Сам парсинг:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Ищем теги модели:
        for selector in possible_selectors:
            found_tags = soup.select(selector)
            if found_tags:
                tags.extend([tag.text.strip().lower() for tag in found_tags])
                break
        # Обычно находятся в div с классами tags, model-tags и т.п.

        # Если тегов не найдено, ищем ключевые слова в тексте страницы:
        if not tags:
            page_text = soup.get_text().lower()
            for tag, possible_keywords in tag2possible_keywords.items():
                if any(keyword in page_text for keyword in possible_keywords):
                    tags.append(tag)

    except requests.exceptions.RequestException:
        pass

    return tags


def model_name2type(model_name: str) -> str:
    """Определяет тип модели по её имени, через парсинг страницы в библиотеке Ollama.

    Возвращает:
        'embeddings' - для моделей эмбеддинга;
        'completions' - для кодогенерирующих моделей;
        'chat' - для диалоговых моделей;
        'unknown' - если не удалось определить.
    """
    # 1. Очистка имени модели для URL
    base_name = re.split(r':\d+[bB]|:latest|:q\d+_K_\w+', model_name)[0]
    base_name = base_name.rstrip(':')
    # Убирает версию после двоеточия (например, ':7b', ':latest', ':q4_K_M').

    # 2. Парсим HTML и собираем теги:
    url = f'https://ollama.com/library/{base_name}'
    tags = url2tags(url)

    # 3. Классификация на основе собранных тегов:
    tag_set = set(tags)
    for type_, possible_tags in type2possible_tags.items():
        if any(tag in tag_set for tag in possible_tags):
            return type_

    # 4. Дополнительные эвристики на основе имени модели:
    base_name = base_name.lower()
    if 'embed' in base_name:
        return 'embeddings'
    if 'code' in base_name:
        return 'completions'
    if 'chat' in base_name:
        return 'chat'

    return 'unknown'


def host2models_info(host: str) -> list[dict]:
    """Получаем список описаний доступных на Ollama моделей."""
    url = f'{host}/api/tags'
    return requests.get(url, timeout=10).json()['models']


def hosts2chat_embd_cmpl_models(
    hosts: Hosts = None,
) -> tuple[Fields, Fields, Fields]:
    """Составляет словари моделей, используемых в конфиг-файле jupyter-ai.

    Разделяет на:
        - словари диалоговых моделей (chat);
        - эмбеддинг-модели (embeddings);
        - модели дополнения (completions).
    """
    # Если список не задан - пытаемся брать имя сервера из окружения:
    if hosts is None:
        hosts = [os.environ[env_var_host]] if env_var_host in os.environ else []

    chat_models: dict[str, dict[str, str]] = {}
    embd_models: dict[str, dict[str, str]] = {}
    cmpl_models: dict[str, dict[str, str]] = {}

    # Перебираем все сервера:
    for host in hosts:
        # Перебираем доступные модели:
        for model_info in host2models_info(host):
            # Определяем имя и типмодели:
            model_name = model_info['name']
            model_type = model_name2type(model_name)

            # Определяем соответствующий словарь:
            if model_type == 'chat':
                cur_dict = chat_models
            elif model_type == 'embeddings':
                cur_dict = embd_models
            elif model_type == 'completions':
                cur_dict = cmpl_models
            else:
                continue

            # Вносим запись в соответствующий словарь, если надо:
            if model_name not in cur_dict:
                cur_dict['ollama:' + model_name] = {'base_url': host}

    return chat_models, embd_models, cmpl_models


def _first_model(fields: Fields) -> str | None:
    """Берёт имя первой модели из словаря."""
    return next(iter(fields), None)


def set_jupyter_ai_settings(
    hosts: Hosts = None,
) -> str:
    """Настраивает Jupyter AI на подключение к Ollama-серверам.

    Аргументы:
        hosts: Адреса сервера Ollama. По-умолчанию используется значение переменной
                    окружения OLLAMA_HOST.

    Возвращает путь до конфигурационного файла.
    """
    if not JUPYTER_AI_AVAILABLE:
        return ''

    # Инициируем содержимое конфигурационного файла:
    cfg_path = Path(jupyter_ai.config_manager.DEFAULT_CONFIG_PATH)
    cfg = (
        json2obj(cfg_path)
        if cfg_path.is_file()
        else {
            'model_provider_id': None,
            'embeddings_provider_id': None,
            'send_with_shift_enter': False,
            'fields': {},
            'api_keys': {},
            'completions_model_provider_id': None,
            'completions_fields': {},
            'embeddings_fields': {},
        }
    )
    # Читаем содержимое из файла или создаём с нуля.

    # Определяем доступные модели:
    chat_models, embd_models, cmpl_models = hosts2chat_embd_cmpl_models(hosts)

    # Собираем словари моделей для конфигураций:
    fields = {**chat_models, **embd_models}
    embeddings_fields = {**embd_models, **chat_models}
    completions_fields = cmpl_models

    # Меняем параметры конфигурации:
    cfg['fields'] = fields  # Диалоговые модели
    cfg['model_provider_id'] = _first_model(fields)
    cfg['embeddings_fields'] = embeddings_fields  # Эмбеддинговые модели
    cfg['embeddings_provider_id'] = _first_model(embeddings_fields)
    cfg['completions_fields'] = completions_fields  # Дополняющие модели
    cfg['completions_model_provider_id'] = _first_model(completions_fields)
    cfg['send_with_shift_enter'] = True  # Отправка сообщение через Shift + Enter

    # Сохраняем файл конфигурации:
    mkdirs(cfg_path.parent)  # Создаём папки, если надо
    return obj2json(cfg, cfg_path)


if __name__ == '__main__':
    set_jupyter_ai_settings()


__all__ = [
    'env_var_host',
    'host2models_info',
    'hosts2chat_embd_cmpl_models',
    'model_name2type',
    'set_jupyter_ai_settings',
]
