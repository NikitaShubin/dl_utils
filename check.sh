#!/bin/bash

set -e  # Выход при первой ошибке

# Цвета для вывода:
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Функция для получения ширины терминала:
get_terminal_width() {
    echo $(tput cols 2>/dev/null || echo 80)
}

# Функция для красивого вывода разделителя:
print_separator() {
    local text="$1"
    local color="${2:-$BLUE}"  # По умолчанию синий цвет
    local width=$(get_terminal_width)
    local text_length=${#text}
    local padding=$(( (width - text_length - 4) / 2 ))  # -4 для учета пробелов и символов

    echo  # Пустая строка перед разделителем

    # Верхняя линия:
    printf "%${width}s\n" | tr ' ' '='

    # Текст с выравниванием по центру:
    if [ $padding -gt 0 ]; then
        printf "%${padding}s ${color}%s${NC} %${padding}s\n" "" "$text" ""
    else
        # Если текст слишком длинный, выводим без отступов:
        printf " ${color}%s${NC} \n" "$text"
    fi

    printf "%${width}s\n" | tr ' ' '='  # Нижняя линия

    echo  # Пустая строка после разделителя
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

# Основной скрипт:
clear
ruff clean  # Очистка кеша Ruff
echo -e "${GREEN}🚀 Запуск проверок качества кода и тестов...${NC}"

# Основные файлы для проверки:
ROOT_FILES=("labels.py" "pt_utils.py")

# Проверка файлов в корне:
for file in "${ROOT_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        print_separator "Ruff format: $file" "$CYAN"
        print_step "Форматирование файла $file..."
        if ruff format "$file"; then
            print_success "Форматирование $file завершено"
        else
            print_error "Ошибка при форматировании $file"
            exit 1
        fi

        print_separator "Ruff check: $file" "$CYAN"
        print_step "Проверка файла $file..."
        if ruff check "$file"; then
            print_success "Проверка $file завершена"
        else
            print_error "Найдены проблемы в $file"
            exit 1
        fi
    else
        print_warning "Файл $file не найден, пропускаем"
    fi
done

# Запуск тестов:
print_separator "Запуск тестов" "$YELLOW"
print_step "Запуск pytest с детализированным выводом..."
if pytest -v; then
    print_success "Все тесты прошли успешно"
else
    print_error "Некоторые тесты не прошли"
    exit 1
fi

# Проверка папки tests:
if [[ -d "tests" ]]; then
    print_separator "Ruff format: tests" "$MAGENTA"
    print_step "Форматирование тестов..."
    if ruff format tests; then
        print_success "Форматирование тестов завершено"
    else
        print_error "Ошибка при форматировании тестов"
        exit 1
    fi

    print_separator "Ruff check: tests" "$MAGENTA"
    print_step "Проверка тестов..."
    if ruff check tests; then
        print_success "Проверка тестов завершена"
    else
        print_error "Найдены проблемы в тестах"
        exit 1
    fi
else
    print_warning "Папка tests не найдена, пропускаем"
fi

# Финальное сообщение:
print_separator "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!" "$GREEN"
echo -e "${GREEN}🎉🎉🎉 Поздравляем! Все проверки завершены успешно! 🎉🎉🎉${NC}"
