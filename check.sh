#!/bin/bash

set -e  # Выход при первой ошибке

echo "🚀 Запуск проверок качества кода и тестов..."

# Основные файлы для проверки:
ROOT_FILES=("labels.py" "pt_utils.py")

# Создаем разделитель
SEPARATOR="============================================================"

# Проверка файлов в корне:
for file in "${ROOT_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo -e "\n$SEPARATOR"
        echo "Ruff format: $file"
        echo "$SEPARATOR"
        ruff format "$file"
        
        echo -e "\n$SEPARATOR"
        echo "Ruff check: $file"
        echo "$SEPARATOR"
        ruff check "$file"
    fi
done

# Запуск тестов:
echo -e "\n$SEPARATOR"
echo "Запуск тестов"
echo "$SEPARATOR"
pytest -v

# Проверка папки tests:
if [[ -d "tests" ]]; then
    echo -e "\n$SEPARATOR"
    echo "Ruff format: tests"
    echo "$SEPARATOR"
    ruff format tests
    
    echo -e "\n$SEPARATOR"
    echo "Ruff check: tests"
    echo "$SEPARATOR"
    ruff check tests
fi

echo -e "\n🎉🎉🎉 Все проверки пройдены успешно! 🎉🎉🎉"