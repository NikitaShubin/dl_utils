#!/bin/bash

# Тонкая точка входа в Python-реализацию py-конвейера чекера. Вся логика -
# в пакете checker/ (pycheck.py); здесь - только окружение для python3 -m:
# путь к пакету на PYTHONPATH и корень dl_utils (конфиги линтеров), чтобы
# CLI и коды возврата остались прежними:
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export DLUTILS_DIR="$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m checker.pycheck "$@"