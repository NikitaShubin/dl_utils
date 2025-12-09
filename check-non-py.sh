#!/bin/bash

# set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

print_separator() {
    echo
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# Получаем абсолютный путь к директории скрипта
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Функция для выполнения проверки
run_check() {
    local description="$1"
    local lint_cmd="$2"
    local find_pattern="$3"
    
    print_separator "Проверка $description"
    
    found_files=0
    while IFS= read -r -d '' file; do
        found_files=$((found_files + 1))
        # Подсвечиваем путь к файлу (относительно корня проекта)
        echo -e "${CYAN}▸ ${MAGENTA}$(realpath --relative-to="$SCRIPT_DIR" "$file")${NC}"
        
        # Запускаем линтер из директории файла с указанием конфига
        ( cd "$(dirname "$file")" && eval "$lint_cmd \"$(basename "$file")\"" )
    done < <(eval "cd \"$SCRIPT_DIR\" && find . -type f $find_pattern \
        ! -path \"./.git/*\" \
        ! -path \"./venv/*\" \
        ! -path \"./.venv/*\" \
        ! -path \"./node_modules/*\" \
        ! -path \"./dist/*\" \
        ! -path \"./build/*\" \
        ! -path \"*/__pycache__/*\" \
        -print0 2>/dev/null" || true)
    
    if [ $found_files -eq 0 ]; then
        echo -e "${GRAY}ℹ️  Файлы не найдены${NC}"
    else
        print_success "Проверка завершена ($found_files файлов)"
    fi
}

# Проверка Dockerfile
echo -e "${BLUE}📁 Конфигурационный файл: $SCRIPT_DIR/.hadolint.yaml${NC}"
run_check "Dockerfile" "hadolint --config \"$SCRIPT_DIR/.hadolint.yaml\"" \
  "\( -name Dockerfile -o -name '*.Dockerfile' \)"

# Проверка shell-скриптов
echo -e "${BLUE}📁 Конфигурационный файл: $SCRIPT_DIR/.shellcheckrc${NC}"
run_check "shell-скрипты" "shellcheck --source-path=\"$SCRIPT_DIR\"" "-name '*.sh'"

# Проверка Markdown файлов
# echo -e "${BLUE}📁 Конфигурационный файл: $SCRIPT_DIR/.markdownlint.yaml${NC}"
# run_check "Markdown файлы" "markdownlint --config \"$SCRIPT_DIR/.markdownlint.yaml\"" "-name '*.md'"

print_separator "ВСЕ ПРОВЕРКИ ЗАВЕРШЕНЫ"
print_success "Все проверки прошли успешно!"