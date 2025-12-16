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

# Получаем абсолютный путь к директории скрипта:
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Проверяем Git-репозиторий относительно директории скрипта
if ! git -C "$SCRIPT_DIR" rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Этот скрипт должен запускаться внутри Git-репозитория"
    exit 1
fi

# Получаем корень Git-репозитория (относительно директории скрипта):
GIT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

# Функция для получения списка файлов из Git-индекса
get_git_files() {
    local pattern="$1"
    # Используем git -C для работы с репозиторием без изменения текущей директории
    git -C "$GIT_ROOT" ls-files "$pattern" 2>/dev/null || true
}

# Функция для рекурсивного поиска файлов Dockerfile
get_dockerfiles() {
    # Рекурсивно ищем все Dockerfile и .Dockerfile файлы
    git -C "$GIT_ROOT" ls-files | grep -E '(/|^)Dockerfile$' || true
    git -C "$GIT_ROOT" ls-files | grep -E '\.Dockerfile$' || true
}

# Функция для выполнения проверки
run_check() {
    local description="$1"
    local lint_cmd="$2"
    shift 2
    local patterns=("$@")

    print_separator "Проверка $description"

    found_files=0
    all_files=()

    # Для Dockerfile используем специальную функцию
    if [[ "$description" == "Dockerfile" ]]; then
        while IFS= read -r file; do
            if [[ -n "$file" && -f "$GIT_ROOT/$file" ]]; then
                all_files+=("$file")
            fi
        done < <(get_dockerfiles)
    else
        # Для остальных файлов используем паттерны
        for pattern in "${patterns[@]}"; do
            while IFS= read -r file; do
                if [[ -n "$file" && -f "$GIT_ROOT/$file" ]]; then
                    all_files+=("$file")
                fi
            done < <(get_git_files "$pattern")
        done
    fi

    # Убираем дубликаты (на случай если файл попал под несколько паттернов)
    if [[ ${#all_files[@]} -gt 0 ]]; then
        mapfile -t all_files < <(printf "%s\n" "${all_files[@]}" | sort -u)
    fi

    # Проверяем каждый файл
    for file in "${all_files[@]}"; do
        found_files=$((found_files + 1))
        echo -e "${CYAN}▸ ${MAGENTA}$file${NC}"

        # Получаем абсолютный путь к файлу
        local file_path
        local file_dir
        local file_name

        file_path="$GIT_ROOT/$file"
        file_dir="$(dirname "$file_path")"
        file_name="$(basename "$file_path")"

        # Запускаем линтер из директории файла (в подпроцессе)
        (cd "$file_dir" && eval "$lint_cmd \"$file_name\"")
    done

    if [ $found_files -eq 0 ]; then
        echo -e "${GRAY}ℹ️  Файлы не найдены${NC}"
    else
        print_success "Проверка завершена ($found_files файлов)"
    fi
}

# Проверка Dockerfile:
echo -e "${BLUE}📁 Конфигурационный файл: $SCRIPT_DIR/.hadolint.yaml${NC}"
run_check "Dockerfile" "hadolint --config \"$SCRIPT_DIR/.hadolint.yaml\""

# Проверка shell-скриптов:
echo -e "${BLUE}📁 Конфигурационный файл: $SCRIPT_DIR/.shellcheckrc${NC}"
run_check "shell-скрипты" "shellcheck --source-path=\"$SCRIPT_DIR\"" "**/*.sh"

# Проверка Markdown файлов:
echo -e "${BLUE}📁 Конфигурационный файл: $SCRIPT_DIR/.markdownlint.yaml${NC}"
run_check "Markdown файлы" "markdownlint --config \"$SCRIPT_DIR/.markdownlint.yaml\"" "**/*.md"

print_separator "ВСЕ ПРОВЕРКИ ЗАВЕРШЕНЫ"
print_success "Все проверки прошли успешно!"