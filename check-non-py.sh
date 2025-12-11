#!/bin/bash

set -e

# Цвета для вывода:
RED='\033[0;31m'
GREEN='\033[0;32m'
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

# Проверяем, что это Git-репозиторий
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Этот скрипт должен запускаться внутри Git-репозитория"
    exit 1
fi

# Получаем абсолютный путь к директории скрипта:
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Получаем корень Git-репозитория:
GIT_ROOT="$(git rev-parse --show-toplevel)"

# Функция для выполнения проверки (принимает произвольное количество паттернов)
run_check() {
    local description="$1"
    local lint_cmd="$2"
    shift 2
    local patterns=("$@")

    print_separator "Проверка $description"

    found_files=0

    # Для каждого паттерна запускаем git ls-files
    for pattern in "${patterns[@]}"; do
        while IFS= read -r file; do
            # Проверяем, что файл существует (на случай удаленных файлов в индексе)
            if [ -f "$GIT_ROOT/$file" ]; then
                found_files=$((found_files + 1))
                echo -e "${CYAN}▸ ${MAGENTA}$file${NC}"
                
                # Запускаем линтер из директории файла
                ( cd "$(dirname "$GIT_ROOT/$file")" && eval "$lint_cmd \"$(basename "$file")\"" )
            fi
        done < <(cd "$GIT_ROOT" && git ls-files "$pattern" 2>/dev/null; exit 0)
    done

    if [ $found_files -eq 0 ]; then
        echo -e "${GRAY}ℹ️  Файлы не найдены${NC}"
    else
        print_success "Проверка завершена ($found_files файлов)"
    fi
}

# Проверка Dockerfile:
echo -e "${BLUE}📁 Конфигурационный файл: $SCRIPT_DIR/.hadolint.yaml${NC}"
run_check "Dockerfile" "hadolint --config \"$SCRIPT_DIR/.hadolint.yaml\"" "Dockerfile" "*.Dockerfile"

# Проверка shell-скриптов:
echo -e "${BLUE}📁 Конфигурационный файл: $SCRIPT_DIR/.shellcheckrc${NC}"
run_check "shell-скрипты" "shellcheck --source-path=\"$SCRIPT_DIR\"" "*.sh"

# Проверка Markdown файлов:
echo -e "${BLUE}📁 Конфигурационный файл: $SCRIPT_DIR/.markdownlint.yaml${NC}"
run_check "Markdown файлы" "markdownlint --config \"$SCRIPT_DIR/.markdownlint.yaml\"" "*.md"

print_separator "ВСЕ ПРОВЕРКИ ЗАВЕРШЕНЫ"
print_success "Все проверки прошли успешно!"