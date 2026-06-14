#!/bin/bash

set -e

# Цвета для вывода:
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[1;34m'
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

# Определяем целевую директорию: либо аргумент, либо директория скрипта
TARGET_DIR="${1:-$SCRIPT_DIR}"

# Преобразуем относительный путь в абсолютный, если это не абсолютный путь
if [[ ! "$TARGET_DIR" = /* ]]; then
    TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"
fi

# Убираем возможный завершающий слэш для корректной работы git -C
TARGET_DIR="${TARGET_DIR%/}"

echo -e "${BLUE}🎯 Целевая директория: $TARGET_DIR${NC}"
echo

# Проверяем, существует ли целевая директория
if [[ ! -d "$TARGET_DIR" ]]; then
    print_error "Целевая директория не существует: $TARGET_DIR"
    exit 1
fi

# Проверяем Git-репозиторий относительно целевой директории
if ! git -C "$TARGET_DIR" rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Целевая директория должна находиться внутри Git-репозитория"
    echo "Запуск из: $TARGET_DIR"
    exit 1
fi

# Получаем корень Git-репозитория (относительно целевой директории):
GIT_ROOT="$(git -C "$TARGET_DIR" rev-parse --show-toplevel)"

echo -e "${BLUE}📦 Корень Git-репозитория: $GIT_ROOT${NC}"
echo

# Функция для получения списка файлов по расширению (из git-индекса и untracked):
get_git_files_by_extension() {
    local ext="$1"
    git -C "$GIT_ROOT" ls-files -c -o --exclude-standard -- "*.$ext" 2>/dev/null || true
}

# Функция для рекурсивного поиска файлов Dockerfile
get_dockerfiles() {
    git -C "$GIT_ROOT" ls-files -c -o --exclude-standard -- "Dockerfile" "*.Dockerfile" 2>/dev/null || true
}

# Функция для поиска docker-compose файлов
get_docker_compose_files() {
    git -C "$GIT_ROOT" ls-files -c -o --exclude-standard -- \
        "docker-compose*.yml" "docker-compose*.yaml" \
        "compose*.yml" "compose*.yaml" 2>/dev/null || true
}

# Функция для выполнения проверки
run_check() {
    local description="$1"
    local lint_cmd="$2"
    local config_file="$3"
    local get_files_func="$4"  # Функция для получения файлов

    print_separator "Проверка $description"
    
    if [[ -n "$config_file" ]]; then
        echo -e "${BLUE}📁 Конфигурационный файл: $config_file${NC}"
        echo
    fi

    found_files=0
    all_files=()

    # Получаем файлы
    while IFS= read -r file; do
        if [[ -n "$file" && -f "$GIT_ROOT/$file" ]]; then
            all_files+=("$file")
        fi
    done < <($get_files_func)

    # Убираем дубликаты
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

        # Запускаем линтер из директории файла
        (cd "$file_dir" && eval "$lint_cmd \"$file_name\"")
    done

    if [ $found_files -eq 0 ]; then
        echo -e "${GRAY}ℹ️  Файлы не найдены${NC}"
    else
        print_success "Проверка завершена ($found_files файлов)"
    fi
}

# Проверка docker-compose файлов:
cfg="$SCRIPT_DIR/.dclintrc"
run_check "🐙 docker-compose файлы" "dclint -c \"$cfg\"" "$cfg" "get_docker_compose_files"

# Проверка Dockerfile:
cfg="$SCRIPT_DIR/.hadolint.yaml"
run_check "🐋 Dockerfile" "hadolint --config \"$cfg\"" "$cfg" "get_dockerfiles"

# Проверка shell-скриптов:
cfg="$SCRIPT_DIR/.shellcheckrc"
run_check "🐚 shell-скрипты" "shellcheck --source-path=\"$cfg\"" "$cfg" "get_git_files_by_extension sh"

# Проверка Markdown файлов:
cfg="$SCRIPT_DIR/.markdownlint.yaml"
run_check "📖 Markdown файлы" "markdownlint --config \"$cfg\"" "$cfg" "get_git_files_by_extension md"

print_separator "ВСЕ ПРОВЕРКИ ЗАВЕРШЕНЫ"
print_success "Все проверки прошли успешно!"