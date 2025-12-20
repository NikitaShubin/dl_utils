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
    Fields,
    Hosts,
    _first_model,
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
            ]
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

            assert 'ollama:llama2:7b' in chat
            assert chat['ollama:llama2:7b']['base_url'] == 'http://localhost:11434'

            assert 'ollama:codellama:latest' in cmpl
            assert 'ollama:nomic-embed-text:latest' in embd

            assert 'ollama:unknown-model:latest' not in chat
            assert 'ollama:unknown-model:latest' not in embd
            assert 'ollama:unknown-model:latest' not in cmpl

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

            assert 'ollama:llama2:7b' in chat
            assert 'ollama:codellama:latest' in cmpl
            assert 'ollama:nomic-embed-text:latest' in embd

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
        chat_models = {'ollama:llama2:7b': {'base_url': 'http://localhost:11434'}}
        embd_models = {
            'ollama:nomic-embed-text:latest': {'base_url': 'http://localhost:11434'}
        }
        cmpl_models = {
            'ollama:codellama:latest': {'base_url': 'http://localhost:11434'}
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
                'existing_completions': {'base_url': 'http://old:11434'}
            },
            'embeddings_fields': {
                'existing_embeddings': {'base_url': 'http://old:11434'}
            },
        }

        chat_models = {'ollama:llama2:7b': {'base_url': 'http://localhost:11434'}}
        embd_models = {
            'ollama:nomic-embed-text:latest': {'base_url': 'http://localhost:11434'}
        }
        cmpl_models = {
            'ollama:codellama:latest': {'base_url': 'http://localhost:11434'}
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


# ============================================================================
# Тесты для типов
# ============================================================================


class TestTypeAnnotations:
    """Тесты для проверки аннотаций типов."""

    def test_fields_type_alias(self) -> None:
        """Тест псевдонима типа Fields."""
        sample_fields: Fields = {
            'ollama:llama2:7b': {'base_url': 'http://localhost:11434'}
        }

        assert isinstance(sample_fields, dict)
        assert 'ollama:llama2:7b' in sample_fields
        assert sample_fields['ollama:llama2:7b']['base_url'] == 'http://localhost:11434'

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
