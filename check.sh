#!/bin/bash

# Файлы из корня репозитория, которые будут проверяться:
root_files=("labels.py" "pt_utils.py" "ollm_utils.py" "boxmot_utils.py" "ul_utils.py" "sam3al.py")

set -e  # Выход при первой ошибке

# Цвета для вывода:
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
PURPLE='\033[0;95m'
NC='\033[0m' # No Color

# Параметры для mypy:
MYPY_ARGS=("--no-incremental" "--show-error-codes" "--warn-unused-ignores" "--follow-imports=skip")

# Функция для получения ширины терминала:
get_terminal_width() {
    tput cols 2>/dev/null || echo 80
}

# Функция для красивого вывода разделителя:
print_separator() {
    local text="$1"
    local color="${2:-$BLUE}"  # По умолчанию синий цвет
    local width
    width=$(get_terminal_width)
    local text_length=${#text}
    local padding=$(( (width - text_length - 4) / 2 ))

    echo
    printf "%${width}s\n" | tr ' ' '='

    if [ $padding -gt 0 ]; then
        printf "%${padding}s ${color}%s${NC} %${padding}s\n" "" "$text" ""
    else
        printf " ${color}%s${NC} \n" "$text"
    fi

    printf "%${width}s\n" | tr ' ' '='
    echo
}

# Функции для цветного вывода:
print_info() {
    echo -e "${BLUE}ℹ️  INFO:${NC} $1"
}
print_success() {
    echo -e "${GREEN}✅ SUCCESS:${NC} $1"
}
print_warning() {
    echo -e "${YELLOW}⚠️  WARNING:${NC} $1"
}
print_error() {
    echo -e "${RED}❌ ERROR:${NC} $1"
}
print_step() {
    echo -e "${CYAN}🔹 $1${NC}"
}

# Получаем абсолютный путь к директории скрипта:
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Проверяем Git-репозиторий относительно директории скрипта
if ! git -C "$SCRIPT_DIR" rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Этот скрипт должен запускаться внутри Git-репозитория"
    exit 1
fi

# Получаем корень Git-репозитория (относительно директории скрипта):
GIT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

# Сбор аргументов для ruff check:
RUFF_CHECK_ARGS=("$@")

# Функция для проверки индексированных файлов
check_indexed_files() {
    local description="$1"
    local check_cmd="$2"
    shift 2
    local patterns=("$@")

    print_separator "$description" "$CYAN"

    found_files=0
    all_files=()

    # Собираем все индексированные файлы по паттернам
    for pattern in "${patterns[@]}"; do
        while IFS= read -r file; do
            if [[ -n "$file" && -f "$GIT_ROOT/$file" ]]; then
                all_files+=("$file")
            fi
        done < <(git -C "$GIT_ROOT" ls-files "$pattern" 2>/dev/null || true)
    done

    # Убираем дубликаты
    if [[ ${#all_files[@]} -gt 0 ]]; then
        mapfile -t all_files < <(printf "%s\n" "${all_files[@]}" | sort -u)
    fi

    # Проверяем каждый файл
    for file in "${all_files[@]}"; do
        found_files=$((found_files + 1))
        print_step "Проверка файла: $file"

        local file_path="$GIT_ROOT/$file"
        local file_dir
        local file_name

        file_dir="$(dirname "$file_path")"
        file_name="$(basename "$file_path")"

        # Запускаем проверку из директории файла
        (cd "$file_dir" && eval "$check_cmd \"$file_name\"")
    done

    if [ $found_files -eq 0 ]; then
        print_warning "Файлы не найдены"
    else
        print_success "Проверка завершена ($found_files файлов)"
    fi
}

# Основной скрипт:
clear
echo -e "${GREEN}🚀 Запуск проверок качества кода и тестов...${NC}"

# Очистка кеша Ruff (из корня репозитория)
print_step "Очистка кеша Ruff..."
ruff clean

# Основные файлы для проверки (только индексированные):
print_separator "Проверка индексированных Python файлов" "$BLUE"

# Проверка каждого файла отдельно
for file in "${root_files[@]}"; do
    # Проверяем, индексирован ли файл
    if git -C "$GIT_ROOT" ls-files --error-unmatch "$file" >/dev/null 2>&1; then
        if [[ -f "$GIT_ROOT/$file" ]]; then
            # Ruff format:
            print_separator "Ruff format: $file" "$CYAN"
            print_step "Форматирование файла $file..."
            (cd "$GIT_ROOT" && ruff format "$file") && print_success "Форматирование $file завершено"

            # Ruff check:
            print_separator "Ruff check: $file" "$CYAN"
            print_step "Проверка файла $file..."
            (cd "$GIT_ROOT" && ruff check "${RUFF_CHECK_ARGS[@]}" "$file") && print_success "Проверка $file завершена"

            # Mypy проверка:
            print_separator "Mypy: $file" "$PURPLE"
            print_step "Проверка типов в файле $file..."
            (cd "$GIT_ROOT" && mypy "${MYPY_ARGS[@]}" "$file") && print_success "Проверка типов $file завершена"
        fi
    else
        print_warning "Файл $file не индексирован или не найден, пропускаем"
    fi
done

# Запуск тестов:
print_separator "Запуск тестов" "$YELLOW"
print_step "Запуск pytest с детализированным выводом..."
(cd "$GIT_ROOT" && pytest -v) && print_success "Все тесты прошли успешно"

# Проверка папки tests (только индексированные файлы):
if [[ -d "$GIT_ROOT/tests" ]]; then
    # Получаем список индексированных .py файлов в папке tests
    test_files=()
    while IFS= read -r file; do
        if [[ -n "$file" && -f "$GIT_ROOT/$file" ]]; then
            test_files+=("$file")
        fi
    done < <(git -C "$GIT_ROOT" ls-files "tests/*.py" 2>/dev/null || true)

    if [ ${#test_files[@]} -gt 0 ]; then
        # Форматируем пути для отображения (убираем префикс tests/)
        display_files=()
        for file in "${test_files[@]}"; do
            display_files+=("${file#tests/}")
        done

        # Ruff format для tests:
        print_separator "Ruff format: tests" "$MAGENTA"
        print_step "Форматирование тестовых файлов: ${display_files[*]}..."
        (cd "$GIT_ROOT" && ruff format tests) && print_success "Форматирование тестов завершено"

        # Ruff check для tests:
        print_separator "Ruff check: tests" "$MAGENTA"
        print_step "Проверка тестов..."
        (cd "$GIT_ROOT" && ruff check "${RUFF_CHECK_ARGS[@]}" tests) && print_success "Проверка тестов завершена"

        # Mypy проверка для tests:
        print_separator "Mypy: tests" "$PURPLE"
        print_step "Проверка типов в тестах..."
        (cd "$GIT_ROOT" && mypy "${MYPY_ARGS[@]}" tests) && print_success "Проверка типов тестов завершена"
    else
        print_warning "В папке tests не найдено индексированных .py файлов"
    fi
else
    print_warning "Папка tests не найдена, пропускаем"
fi

# Финальное сообщение:
print_separator "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!" "$GREEN"
echo -e "${GREEN}🎉🎉🎉 Поздравляем! Все проверки завершены успешно! 🎉🎉🎉${NC}"