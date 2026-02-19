"""Тесты для модуля sam3al (загрузка SAM3)."""

from pathlib import Path
from unittest.mock import patch

import pytest

import sam3al
from sam3al import download_sam3


@pytest.fixture
def mock_path_home(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Заменяет Path.home() на фиксированный путь."""
    mock_home = Path('/fake/home')
    monkeypatch.setattr(Path, 'home', lambda: mock_home)
    return mock_home


class TestDownloadSam3:
    """Тесты для функции download_sam3."""

    def test_none_path_file_exists(
        self,
        mock_path_home: Path,
    ) -> None:
        """path=None, файл существует -> возвращает ~/models/sam3.pt."""
        expected_path = mock_path_home / 'models' / 'sam3.pt'

        with patch.object(Path, 'exists', return_value=True) as mock_exists:
            result = download_sam3(None)

        assert result == expected_path
        mock_exists.assert_called_once()

    def test_none_path_file_does_not_exist(
        self,
        mock_path_home: Path,
    ) -> None:
        """path=None, файла нет -> скачивает и возвращает ~/models/sam3.pt."""
        expected_path = mock_path_home / 'models' / 'sam3.pt'
        temp_dir = expected_path.parent / '.temp_download'
        fake_downloaded_file = temp_dir / 'sam3.pt'

        with (
            patch.object(Path, 'exists', return_value=False) as mock_exists,
            patch.object(Path, 'mkdir') as mock_mkdir,
            patch('shutil.move') as mock_move,
            patch('shutil.rmtree') as mock_rmtree,
            patch('sam3al.model_file_download') as mock_download,
        ):
            mock_download.return_value = str(fake_downloaded_file)
            result = download_sam3(None)

        assert result == expected_path
        mock_exists.assert_called_once()
        mock_download.assert_called_once_with(
            model_id='facebook/sam3',
            file_path='sam3.pt',
            local_dir=temp_dir,
        )
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=False)
        mock_move.assert_called_once_with(
            str(fake_downloaded_file),
            expected_path,
        )
        mock_rmtree.assert_called_once_with(temp_dir)

    def test_path_with_pt_suffix_exists(self) -> None:
        """Path заканчивается на .pt, файл существует -> возвращается path."""
        path = Path('/some/path/model.pt')
        with patch.object(Path, 'exists', return_value=True) as mock_exists:
            result = download_sam3(path)

        assert result == path
        mock_exists.assert_called_once()

    def test_path_with_pt_suffix_not_exists(self) -> None:
        """Path заканчивается на .pt, файла нет -> скачивается в этот путь."""
        path = Path('/some/path/model.pt')
        temp_dir = path.parent / '.temp_download'
        fake_downloaded = temp_dir / 'sam3.pt'

        with (
            patch.object(Path, 'exists', return_value=False),
            patch.object(Path, 'mkdir') as mock_mkdir,
            patch('shutil.move') as mock_move,
            patch('shutil.rmtree') as mock_rmtree,
            patch('sam3al.model_file_download') as mock_download,
        ):
            mock_download.return_value = str(fake_downloaded)
            result = download_sam3(path)

        assert result == path
        mock_download.assert_called_once_with(
            model_id='facebook/sam3',
            file_path='sam3.pt',
            local_dir=temp_dir,
        )
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=False)
        mock_move.assert_called_once_with(
            str(fake_downloaded),
            path,
        )
        mock_rmtree.assert_called_once_with(temp_dir)

    def test_path_without_pt_suffix_exists(self) -> None:
        """Path без .pt, файл существует -> возвращается path/sam3.pt."""
        path = Path('/some/folder')
        expected_path = path / 'sam3.pt'

        with patch.object(Path, 'exists', return_value=True) as mock_exists:
            result = download_sam3(path)

        assert result == expected_path
        mock_exists.assert_called_once()

    def test_path_without_pt_suffix_not_exists(self) -> None:
        """Path без .pt, файла нет -> скачивается в path/sam3.pt."""
        path = Path('/some/folder')
        expected_path = path / 'sam3.pt'
        temp_dir = path / '.temp_download'
        fake_downloaded = temp_dir / 'sam3.pt'

        with (
            patch.object(Path, 'exists', return_value=False),
            patch.object(Path, 'mkdir') as mock_mkdir,
            patch('shutil.move') as mock_move,
            patch('shutil.rmtree') as mock_rmtree,
            patch('sam3al.model_file_download') as mock_download,
        ):
            mock_download.return_value = str(fake_downloaded)
            result = download_sam3(path)

        assert result == expected_path
        mock_download.assert_called_once_with(
            model_id='facebook/sam3',
            file_path='sam3.pt',
            local_dir=temp_dir,
        )
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=False)
        mock_move.assert_called_once_with(
            str(fake_downloaded),
            expected_path,
        )
        mock_rmtree.assert_called_once_with(temp_dir)

    def test_path_as_string(self) -> None:
        """Path передаётся строкой -> корректно преобразуется в Path."""
        path_str = '/some/path/model.pt'
        expected_path = Path(path_str)
        with patch.object(Path, 'exists', return_value=True) as mock_exists:
            result = download_sam3(path_str)

        assert result == expected_path
        mock_exists.assert_called_once()

    def test_download_creates_temp_dir_only_once(self) -> None:
        """Проверяет создание временной папки с правильными параметрами."""
        path = Path('/some/path/model.pt')
        temp_dir = path.parent / '.temp_download'

        with (
            patch.object(Path, 'exists', return_value=False),
            patch.object(Path, 'mkdir') as mock_mkdir,
            patch('shutil.move'),
            patch('shutil.rmtree'),
            patch('sam3al.model_file_download') as mock_download,
        ):
            mock_download.return_value = str(temp_dir / 'sam3.pt')
            download_sam3(path)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=False)

    def test_modelscope_not_imported_if_file_exists(self) -> None:
        """Проверяет, что modelscope не импортируется при наличии файла."""
        with (
            patch.object(Path, 'exists', return_value=True),
            patch('builtins.__import__') as mock_import,
        ):
            download_sam3(None)

            # Убеждаемся, что modelscope не импортировался
            calls = [
                call
                for call in mock_import.call_args_list
                if call[0][0] == 'modelscope'
            ]
            assert not calls

    def test_raises_if_modelscope_missing(self) -> None:
        """Проверяет, что при отсутствии modelscope выбрасывается исключение."""
        # Имитируем отсутствие modelscope, подменяя переменную модуля
        fake_error = ModuleNotFoundError("No module named 'modelscope'")
        with (
            patch.object(sam3al, 'ModelscopeNotFoundError', fake_error),
            patch.object(Path, 'exists', return_value=False),
            pytest.raises(ModuleNotFoundError) as exc_info,
        ):
            download_sam3(Path('/some/path/model.pt'))

        assert exc_info.value is fake_error
