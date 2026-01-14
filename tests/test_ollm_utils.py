"""Тесты для ollm_utils.py.

Модуль содержит тесты для функций работы с Ollama-сервисом.
"""

import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

# Импортируем только если модуль доступен
from ollm_utils import (
    Chat,
    Fields,
    Hosts,
    _first_model,
    _ollama_prefix,
    env_var_host,
    file2context,
    host2models_info,
    hosts2chat_embd_cmpl_models,
    model_name2type,
    set_jupyter_ai_settings,
    url2tags,
)

# ============================================================================
# Тесты для вспомогательных функций
# ============================================================================


class TestHelperFunctions:
    """Тесты для вспомогательных функций."""

    def test_first_model(self) -> None:
        """Тест функции _first_model."""
        # С пустым словарём
        assert _first_model({}) is None

        # С непустым словарём
        fields = {
            'model1': {'base_url': 'http://localhost'},
            'model2': {'base_url': 'http://server2'},
        }
        assert _first_model(fields) == 'model1'

        # С одним элементом
        fields = {'single_model': {'base_url': 'http://localhost'}}
        assert _first_model(fields) == 'single_model'

        # С пустым списком
        assert _first_model([]) is None

        # С непустым списком
        assert _first_model(['model1', 'model2']) == 'model1'


class TestOllamaPrefix:
    """Тесты для функции _ollama_prefix."""

    def test_ollama_prefix(self) -> None:
        """Тест добавления префикса 'ollama:' к ключам словаря."""
        # Пустой словарь
        assert _ollama_prefix({}) == {}

        # Словарь с одним элементом
        models = {'llama2:7b': {'base_url': 'http://localhost:11434'}}
        result = _ollama_prefix(models)
        assert 'ollama:llama2:7b' in result
        assert result['ollama:llama2:7b'] == {'base_url': 'http://localhost:11434'}

        # Словарь с несколькими элементами
        models = {
            'llama2:7b': {'base_url': 'http://localhost:11434'},
            'codellama': {'base_url': 'http://localhost:11434'},
        }
        result = _ollama_prefix(models)
        assert 'ollama:llama2:7b' in result
        assert 'ollama:codellama' in result
        assert len(result) == 2


# ============================================================================
# Тесты для url2tags
# ============================================================================


class TestUrl2Tags:
    """Тесты для функции url2tags."""

    def test_url2tags_with_tags(self) -> None:
        """Тест извлечения тегов из HTML с тегами."""
        html_content = """
        <html>
            <body>
                <div class="tags">
                    <a>chat</a>
                    <a>conversation</a>
                </div>
            </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html_content
        mock_response.raise_for_status.return_value = None

        with patch('ollm_utils.requests.get', return_value=mock_response):
            tags = url2tags('https://ollama.com/library/test-model')
            assert 'chat' in tags
            assert 'conversation' in tags
            assert len(tags) == 2

    def test_url2tags_without_tags(self) -> None:
        """Тест извлечения тегов из HTML без явных тегов."""
        html_content = """
        <html>
            <body>
                <p>This is a model for text embeddings and vector search.</p>
            </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html_content
        mock_response.raise_for_status.return_value = None

        with patch('ollm_utils.requests.get', return_value=mock_response):
            tags = url2tags('https://ollama.com/library/embedding-model')
            assert 'embeddings' in tags
            assert len(tags) == 1

    def test_url2tags_error(self) -> None:
        """Тест обработки ошибки при запросе."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.RequestException('Connection error')
        )

        with patch('ollm_utils.requests.get', return_value=mock_response):
            tags = url2tags('https://ollama.com/library/nonexistent')
            assert tags == []


# ============================================================================
# Тесты для model_name2type
# ============================================================================


class TestModelName2Type:
    """Тесты для функции model_name2type."""

    @pytest.mark.parametrize(
        ('model_name', 'expected_type'),
        [
            ('llama2:7b', 'chat'),
            ('codellama:latest', 'completions'),
            ('nomic-embed-text:latest', 'embeddings'),
            ('llama2:13b:q4_K_M', 'chat'),
            ('unknown-model:latest', 'unknown'),
        ],
    )
    def test_model_name2type_with_mock(
        self,
        model_name: str,
        expected_type: str,
    ) -> None:
        """Тест определения типа модели с моком запроса."""
        with patch('ollm_utils.url2tags') as mock_url2tags:
            if 'llama2' in model_name:
                mock_url2tags.return_value = ['chat', 'conversation']
            elif 'codellama' in model_name:
                mock_url2tags.return_value = ['code', 'programming']
            elif 'nomic-embed-text' in model_name:
                mock_url2tags.return_value = ['embeddings', 'vector']
            else:
                mock_url2tags.return_value = []

            result = model_name2type(model_name)
            assert result == expected_type

    def test_model_name2type_heuristics(self) -> None:
        """Тест эвристик на основе имени модели."""
        with patch('ollm_utils.url2tags', return_value=[]):
            assert model_name2type('embed-model:latest') == 'embeddings'
            assert model_name2type('code-llama:7b') == 'completions'
            assert model_name2type('chat-model:latest') == 'chat'
            assert model_name2type('random-model:latest') == 'unknown'


# ============================================================================
# Тесты для host2models_info
# ============================================================================


class TestHost2ModelsInfo:
    """Тесты для функции host2models_info."""

    def test_host2models_info_success(self) -> None:
        """Тест успешного получения списка моделей."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'models': [
                {'name': 'llama2:7b', 'modified_at': '2023-01-01T00:00:00Z'},
                {'name': 'codellama:latest', 'modified_at': '2023-01-02T00:00:00Z'},
                {
                    'name': 'nomic-embed-text:latest',
                    'modified_at': '2023-01-03T00:00:00Z',
                },
            ],
        }

        with patch('ollm_utils.requests.get', return_value=mock_response):
            models = host2models_info('http://localhost:11434')

            assert len(models) == 3
            assert models[0]['name'] == 'llama2:7b'
            assert models[1]['name'] == 'codellama:latest'
            assert models[2]['name'] == 'nomic-embed-text:latest'

    def test_host2models_info_timeout(self) -> None:
        """Тест таймаута при запросе."""
        with (
            patch('ollm_utils.requests.get', side_effect=Exception('Timeout')),
            pytest.raises(Exception, match='Timeout'),
        ):
            host2models_info('http://localhost:11434')


# ============================================================================
# Тесты для hosts2chat_embd_cmpl_models
# ============================================================================


class TestHosts2ChatEmbdCmplModels:
    """Тесты для функции hosts2chat_embd_cmpl_models."""

    def test_hosts2chat_embd_cmpl_models_with_hosts(self) -> None:
        """Тест с явным указанием хостов."""
        models = [
            {'name': 'llama2:7b'},
            {'name': 'codellama:latest'},
            {'name': 'nomic-embed-text:latest'},
            {'name': 'unknown-model:latest'},
        ]

        type_map = {
            'llama2:7b': 'chat',
            'codellama:latest': 'completions',
            'nomic-embed-text:latest': 'embeddings',
            'unknown-model:latest': 'unknown',
        }

        with (
            patch('ollm_utils.host2models_info', return_value=models),
            patch(
                'ollm_utils.model_name2type',
                side_effect=lambda x: type_map.get(x, 'unknown'),
            ),
        ):
            hosts = ['http://localhost:11434']
            chat, embd, cmpl = hosts2chat_embd_cmpl_models(hosts)

            # Ожидаем модели БЕЗ префикса "ollama:"
            assert 'llama2:7b' in chat
            assert chat['llama2:7b']['base_url'] == 'http://localhost:11434'

            assert 'codellama:latest' in cmpl
            assert 'nomic-embed-text:latest' in embd

            assert 'unknown-model:latest' not in chat
            assert 'unknown-model:latest' not in embd
            assert 'unknown-model:latest' not in cmpl

    def test_hosts2chat_embd_cmpl_models_without_hosts(self) -> None:
        """Тест без указания хостов (используется переменная окружения)."""
        models = [
            {'name': 'llama2:7b'},
            {'name': 'codellama:latest'},
            {'name': 'nomic-embed-text:latest'},
        ]

        type_map = {
            'llama2:7b': 'chat',
            'codellama:latest': 'completions',
            'nomic-embed-text:latest': 'embeddings',
        }

        with (
            patch.dict(
                'ollm_utils.os.environ',
                {'OLLAMA_HOST': 'http://localhost:11434'},
            ),
            patch('ollm_utils.host2models_info', return_value=models),
            patch(
                'ollm_utils.model_name2type',
                side_effect=lambda x: type_map.get(x, 'unknown'),
            ),
        ):
            chat, embd, cmpl = hosts2chat_embd_cmpl_models()

            # Ожидаем модели БЕЗ префикса "ollama:"
            assert 'llama2:7b' in chat
            assert 'codellama:latest' in cmpl
            assert 'nomic-embed-text:latest' in embd

    def test_hosts2chat_embd_cmpl_models_empty_hosts(self) -> None:
        """Тест с пустым списком хостов."""
        with (
            patch.dict(
                'ollm_utils.os.environ',
                {'OLLAMA_HOST': 'http://localhost:11434'},
            ),
            patch('ollm_utils.host2models_info', return_value=[]),
        ):
            chat, embd, cmpl = hosts2chat_embd_cmpl_models([])

            assert chat == {}
            assert embd == {}
            assert cmpl == {}

    def test_hosts2chat_embd_cmpl_models_no_env_var(self) -> None:
        """Тест без переменной окружения и без хостов."""
        with (
            patch.dict('ollm_utils.os.environ', {}, clear=True),
            patch('ollm_utils.host2models_info') as mock_host2models,
        ):
            chat, embd, cmpl = hosts2chat_embd_cmpl_models(None)

            assert chat == {}
            assert embd == {}
            assert cmpl == {}
            mock_host2models.assert_not_called()


# ============================================================================
# Тесты для set_jupyter_ai_settings
# ============================================================================


class TestSetJupyterAiSettings:
    """Тесты для функции set_jupyter_ai_settings."""

    @pytest.fixture
    def temp_config_file(self) -> Iterator[str]:
        """Создаёт временный конфигурационный файл."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    def test_set_jupyter_ai_settings_unavailable(self) -> None:
        """Тест, когда jupyter_ai недоступен."""
        # Мокаем модуль, чтобы его не было
        with patch('ollm_utils.JUPYTER_AI_AVAILABLE', new=False):
            result = set_jupyter_ai_settings()
            assert result == ''

    def test_set_jupyter_ai_settings_success(self, temp_config_file: str) -> None:
        """Тест успешной настройки Jupyter AI."""
        # Моки возвращают модели БЕЗ префикса
        chat_models: Fields = {'llama2:7b': {'base_url': 'http://localhost:11434'}}
        embd_models: Fields = {
            'nomic-embed-text:latest': {'base_url': 'http://localhost:11434'},
        }
        cmpl_models: Fields = {
            'codellama:latest': {'base_url': 'http://localhost:11434'},
        }

        mock_jupyter_ai = Mock()
        mock_jupyter_ai.config_manager.DEFAULT_CONFIG_PATH = temp_config_file

        with (
            patch('ollm_utils.JUPYTER_AI_AVAILABLE', new=True),
            patch('ollm_utils.jupyter_ai', mock_jupyter_ai),
            patch(
                'ollm_utils.hosts2chat_embd_cmpl_models',
                return_value=(chat_models, embd_models, cmpl_models),
            ) as mock_hosts2models,
            patch('ollm_utils.json2obj', return_value={}),
            patch('ollm_utils.obj2json', return_value=temp_config_file),
            patch('ollm_utils.mkdirs') as mock_mkdirs,
        ):
            result = set_jupyter_ai_settings(['http://localhost:11434'])

            assert result == temp_config_file
            mock_hosts2models.assert_called_once_with(['http://localhost:11434'])
            mock_mkdirs.assert_called_once()

    def test_set_jupyter_ai_settings_existing_config(
        self,
        temp_config_file: str,
    ) -> None:
        """Тест с существующим конфигурационном файлом."""
        existing_config = {
            'model_provider_id': 'existing_model',
            'embeddings_provider_id': 'existing_embeddings',
            'completions_model_provider_id': 'existing_completions',
            'send_with_shift_enter': False,
            'fields': {'existing_model': {'base_url': 'http://old:11434'}},
            'api_keys': {},
            'completions_fields': {
                'existing_completions': {'base_url': 'http://old:11434'},
            },
            'embeddings_fields': {
                'existing_embeddings': {'base_url': 'http://old:11434'},
            },
        }

        # Моки возвращают модели БЕЗ префикса
        chat_models: Fields = {'llama2:7b': {'base_url': 'http://localhost:11434'}}
        embd_models: Fields = {
            'nomic-embed-text:latest': {'base_url': 'http://localhost:11434'},
        }
        cmpl_models: Fields = {
            'codellama:latest': {'base_url': 'http://localhost:11434'},
        }

        mock_jupyter_ai = Mock()
        mock_jupyter_ai.config_manager.DEFAULT_CONFIG_PATH = temp_config_file

        with (
            patch('ollm_utils.JUPYTER_AI_AVAILABLE', new=True),
            patch('ollm_utils.jupyter_ai', mock_jupyter_ai),
            patch('ollm_utils.Path.is_file', return_value=True),
            patch('ollm_utils.json2obj', return_value=existing_config),
            patch(
                'ollm_utils.hosts2chat_embd_cmpl_models',
                return_value=(chat_models, embd_models, cmpl_models),
            ),
            patch('ollm_utils.obj2json', return_value=temp_config_file),
            patch('ollm_utils.mkdirs') as mock_mkdirs,
        ):
            result = set_jupyter_ai_settings(['http://localhost:11434'])

            assert result == temp_config_file
            mock_mkdirs.assert_called_once()

    def test_set_jupyter_ai_settings_creates_config_if_not_exists(
        self,
        temp_config_file: str,
    ) -> None:
        """Тест создания конфигурации, если файл не существует."""
        # Моки возвращают модели БЕЗ префикса
        chat_models: Fields = {'llama2:7b': {'base_url': 'http://localhost:11434'}}
        embd_models: Fields = {}
        cmpl_models: Fields = {}

        mock_jupyter_ai = Mock()
        mock_jupyter_ai.config_manager.DEFAULT_CONFIG_PATH = temp_config_file

        with (
            patch('ollm_utils.JUPYTER_AI_AVAILABLE', new=True),
            patch('ollm_utils.jupyter_ai', mock_jupyter_ai),
            patch('ollm_utils.Path.is_file', return_value=False),
            patch(
                'ollm_utils.hosts2chat_embd_cmpl_models',
                return_value=(chat_models, embd_models, cmpl_models),
            ),
            patch('ollm_utils.obj2json') as mock_obj2json,
            patch('ollm_utils.mkdirs') as mock_mkdirs,
        ):
            mock_obj2json.return_value = temp_config_file

            result = set_jupyter_ai_settings(['http://localhost:11434'])

            assert result == temp_config_file
            mock_mkdirs.assert_called_once()


# ============================================================================
# Тесты для file2context
# ============================================================================


class TestFile2Context:
    """Тесты для функции file2context."""

    def test_file2context_with_single_file(self, tmp_path: Path) -> None:
        """Тест с одним файлом."""
        # Создаем временный файл
        test_file = tmp_path / 'test.txt'
        test_file.write_text('Содержимое файла', encoding='utf-8')

        result = file2context(test_file)

        assert 'ФАЙЛ: test.txt' in result
        assert 'Содержимое файла' in result
        assert 'КОНЕЦ ФАЙЛА: test.txt' in result

    def test_file2context_with_file_list(self, tmp_path: Path) -> None:
        """Тест со списком файлов."""
        # Создаем временные файлы
        file1 = tmp_path / 'file1.txt'
        file1.write_text('Содержимое 1', encoding='utf-8')

        file2 = tmp_path / 'file2.txt'
        file2.write_text('Содержимое 2', encoding='utf-8')

        result = file2context([file1, file2])

        assert 'ФАЙЛ: file1.txt' in result
        assert 'Содержимое 1' in result
        assert 'ФАЙЛ: file2.txt' in result
        assert 'Содержимое 2' in result
        assert 'КОНЕЦ ФАЙЛА: file1.txt' in result
        assert 'КОНЕЦ ФАЙЛА: file2.txt' in result

    def test_file2context_with_string_path(self, tmp_path: Path) -> None:
        """Тест с путем в виде строки."""
        # Создаем временный файл
        test_file = tmp_path / 'test.txt'
        test_file.write_text('Содержимое', encoding='utf-8')

        result = file2context(str(test_file))

        assert 'ФАЙЛ: test.txt' in result
        assert 'Содержимое' in result

    def test_file2context_empty_files_list(self) -> None:
        """Тест с пустым списком файлов."""
        result = file2context([])
        assert result == ''


# ============================================================================
# Тесты для класса Chat
# ============================================================================


class TestChat:
    """Тесты для класса Chat."""

    @pytest.fixture
    def chat_instance(self) -> Chat:
        """Создает экземпляр Chat для тестов."""
        with patch.dict(
            'ollm_utils.os.environ',
            {'OLLAMA_HOST': 'http://localhost:11434'},
        ):
            return Chat(host='http://localhost:11434', model='test-model')

    def test_chat_init_with_host_and_model(self) -> None:
        """Тест инициализации Chat с указанием хоста и модели."""
        chat = Chat(host='http://localhost:11434', model='test-model')
        assert chat.host == 'http://localhost:11434'
        assert chat.model == 'test-model'
        assert chat.temperature == 0.0
        assert chat.timeout == 300
        assert chat.seed == 42
        assert chat.messages == []

    def test_chat_init_without_host_uses_env_var(self) -> None:
        """Тест инициализации Chat без указания хоста (используется переменная окружения)."""  # noqa: E501
        with patch.dict(
            'ollm_utils.os.environ',
            {'OLLAMA_HOST': 'http://localhost:11434'},
        ):
            chat = Chat(model='test-model')
            assert chat.host == 'http://localhost:11434'
            assert chat.model == 'test-model'

    def test_chat_init_without_host_no_env_var(self) -> None:
        """Тест инициализации Chat без хоста и переменной окружения."""
        with patch.dict('ollm_utils.os.environ', {}, clear=True):
            chat = Chat(host='', model='test-model')
            assert chat.host == 'http://'
            assert chat.model == 'test-model'

    def test_chat_init_with_host_without_protocol(self) -> None:
        """Тест инициализации Chat с хостом без протокола."""
        chat = Chat(host='localhost:11434', model='test-model')
        assert chat.host == 'http://localhost:11434'

    def test_chat_init_with_https_host(self) -> None:
        """Тест инициализации Chat с HTTPS хостом."""
        chat = Chat(host='https://localhost:11434', model='test-model')
        assert chat.host == 'https://localhost:11434'

    def test_chat_init_without_model(self) -> None:
        """Тест инициализации Chat без указания модели."""
        with (
            patch.dict(
                'ollm_utils.os.environ',
                {'OLLAMA_HOST': 'http://localhost:11434'},
            ),
            patch.object(Chat, 'get_models') as mock_get_models,
            patch('ollm_utils._first_model') as mock_first_model,
        ):
            mock_get_models.return_value = (['model1'], ['model2'], ['model3'])
            mock_first_model.return_value = 'model1'

            chat = Chat()
            assert chat.model == 'model1'

    def test_chat_init_without_model_no_models_found(self) -> None:
        """Тест инициализации Chat без модели, когда нет доступных моделей."""
        with (
            patch.dict(
                'ollm_utils.os.environ',
                {'OLLAMA_HOST': 'http://localhost:11434'},
            ),
            patch.object(Chat, 'get_models') as mock_get_models,
            patch('ollm_utils._first_model') as mock_first_model,
        ):
            mock_get_models.return_value = ([], [], [])
            mock_first_model.return_value = None

            with pytest.raises(ValueError, match='Модель не найдена!'):
                Chat()

    def test_chat_get_models(self, chat_instance: Chat) -> None:
        """Тест метода get_models."""
        with patch('ollm_utils.hosts2chat_embd_cmpl_models') as mock_hosts2models:
            mock_hosts2models.return_value = (
                {'model1': {'base_url': 'http://localhost:11434'}},
                {'model2': {'base_url': 'http://localhost:11434'}},
                {'model3': {'base_url': 'http://localhost:11434'}},
            )

            chat, embd, cmpl = chat_instance.get_models()

            assert chat == ['model1']
            assert embd == ['model2']
            assert cmpl == ['model3']
            mock_hosts2models.assert_called_once_with([chat_instance.host])

    def test_chat_call_success(self, chat_instance: Chat) -> None:
        """Тест успешного вызова модели через __call__."""
        mock_response = Mock()
        mock_response.json.return_value = {'message': {'content': 'Ответ модели'}}
        mock_response.raise_for_status.return_value = None

        with patch('ollm_utils.requests.post', return_value=mock_response):
            result = chat_instance('Привет!')

            assert result == 'Ответ модели'
            assert len(chat_instance.messages) == 2
            assert chat_instance.messages[0]['role'] == 'user'
            assert chat_instance.messages[0]['content'] == 'Привет!'
            assert chat_instance.messages[1]['role'] == 'assistant'
            assert chat_instance.messages[1]['content'] == 'Ответ модели'

    def test_chat_call_with_file(self, chat_instance: Chat, tmp_path: Path) -> None:
        """Тест вызова модели с файлом."""
        # Создаем временный файл
        test_file = tmp_path / 'test.txt'
        test_file.write_text('Содержимое файла', encoding='utf-8')

        mock_response = Mock()
        mock_response.json.return_value = {'message': {'content': 'Ответ модели'}}
        mock_response.raise_for_status.return_value = None

        with patch('ollm_utils.requests.post', return_value=mock_response):
            result = chat_instance('Вопрос!', file=test_file)

            assert result == 'Ответ модели'
            # Проверяем, что содержимое файла добавлено к сообщению
            assert len(chat_instance.messages) == 2
            assert chat_instance.messages[0]['role'] == 'user'
            assert 'Содержимое файла' in chat_instance.messages[0]['content']
            assert 'Вопрос!' in chat_instance.messages[0]['content']

    def test_chat_call_with_file_list(
        self,
        chat_instance: Chat,
        tmp_path: Path,
    ) -> None:
        """Тест вызова модели со списком файлов."""
        # Создаем временные файлы
        file1 = tmp_path / 'file1.txt'
        file1.write_text('Содержимое 1', encoding='utf-8')

        file2 = tmp_path / 'file2.txt'
        file2.write_text('Содержимое 2', encoding='utf-8')

        mock_response = Mock()
        mock_response.json.return_value = {'message': {'content': 'Ответ модели'}}
        mock_response.raise_for_status.return_value = None

        with patch('ollm_utils.requests.post', return_value=mock_response):
            result = chat_instance('Вопрос!', file=[file1, file2])

            assert result == 'Ответ модели'
            assert len(chat_instance.messages) == 2
            assert 'Содержимое 1' in chat_instance.messages[0]['content']
            assert 'Содержимое 2' in chat_instance.messages[0]['content']

    def test_chat_call_with_temperature_zero_warmup(self, chat_instance: Chat) -> None:
        """Тест вызова модели с temperature=0 (должен вызываться warmup)."""
        chat_instance.temperature = 0.0

        # Создаем моки для двух запросов
        mock_response_warmup = Mock()
        mock_response_warmup.raise_for_status.return_value = None

        mock_response_chat = Mock()
        mock_response_chat.json.return_value = {'message': {'content': 'Ответ модели'}}
        mock_response_chat.raise_for_status.return_value = None

        with patch('ollm_utils.requests.post') as mock_post:
            mock_post.side_effect = [mock_response_warmup, mock_response_chat]

            result = chat_instance('Привет!')

            assert result == 'Ответ модели'
            assert mock_post.call_count == 2

    def test_chat_call_with_temperature_non_zero_no_warmup(
        self,
        chat_instance: Chat,
    ) -> None:
        """Тест вызова модели с temperature>0 (не должен вызываться warmup)."""
        chat_instance.temperature = 0.7

        mock_response = Mock()
        mock_response.json.return_value = {'message': {'content': 'Ответ модели'}}
        mock_response.raise_for_status.return_value = None

        with (
            patch('ollm_utils.requests.post', return_value=mock_response),
            patch.object(chat_instance, '_warmup_model') as mock_warmup,
        ):
            result = chat_instance('Привет!')

            assert result == 'Ответ модели'
            mock_warmup.assert_not_called()

    def test_chat_warmup_model(self, chat_instance: Chat) -> None:
        """Тест метода _warmup_model."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        with patch('ollm_utils.requests.post', return_value=mock_response):
            chat_instance._warmup_model()  # noqa: SLF001

            # Проверяем, что запрос был сделан с правильными параметрами
            mock_response.raise_for_status.assert_called_once()

    def test_chat_reset(self, chat_instance: Chat) -> None:
        """Тест метода reset."""
        # Добавляем сообщения в историю
        chat_instance.messages = [
            {'role': 'user', 'content': 'Привет'},
            {'role': 'assistant', 'content': 'Привет!'},
        ]

        assert len(chat_instance.messages) == 2

        chat_instance.reset()

        assert len(chat_instance.messages) == 0

    def test_chat_call_error(self, chat_instance: Chat) -> None:
        """Тест обработки ошибки при вызове модели."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception('API Error')

        with (
            patch('ollm_utils.requests.post', return_value=mock_response),
            pytest.raises(Exception, match='API Error'),
        ):
            chat_instance('Привет!')


# ============================================================================
# Тесты для констант
# ============================================================================


class TestConstants:
    """Тесты для констант."""

    def test_env_var_host(self) -> None:
        """Тест константы env_var_host."""
        assert env_var_host == 'OLLAMA_HOST'


# ============================================================================
# Тесты для типов
# ============================================================================


class TestTypeAnnotations:
    """Тесты для проверки аннотаций типов."""

    def test_fields_type_alias(self) -> None:
        """Тест псевдонима типа Fields."""
        sample_fields: Fields = {
            'llama2:7b': {'base_url': 'http://localhost:11434'},
        }

        assert isinstance(sample_fields, dict)
        assert 'llama2:7b' in sample_fields
        assert sample_fields['llama2:7b']['base_url'] == 'http://localhost:11434'

    def test_hosts_type_alias(self) -> None:
        """Тест псевдонима типа Hosts."""
        hosts_list: Hosts = ['http://localhost:11434']
        hosts_set: Hosts = {'http://localhost:11434'}
        hosts_tuple: Hosts = ('http://localhost:11434',)
        hosts_none: Hosts = None

        assert isinstance(hosts_list, list)
        assert isinstance(hosts_set, set)
        assert isinstance(hosts_tuple, tuple)
        assert hosts_none is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
