#!/bin/bash

# Общий модуль утилит для скриптов dl_utils. Подключается через source.

# Палитра цветов (используются подключающими скриптами):
# shellcheck disable=SC2034
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;95m'
MAGENTA='\033[0;35m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Ширина терминала:
get_terminal_width() {
    tput cols 2>/dev/null || echo 80
}

# Сплошная линия ═ с текстом по центру:
#   ═══════════════════ Текст ═══════════════════
print_separator() {
    local text="$1"
    local color="${2:-$CYAN}"
    local width
    width=$(get_terminal_width)
    local text_length=${#text}
    local total=$((width - 1 - text_length - 2))
    local side=$((total / 2))
    local right_side=$((total - side))
    local pad_l pad_r left right
    pad_l=$(printf "%${side}s" "")
    pad_r=$(printf "%${right_side}s" "")
    left="${pad_l// /═}"
    right="${pad_r// /═}"
    echo
    echo -e "${color}${left} ${text} ${right}${NC}"
    echo
}

# Функции вывода (лейбл цветом, сообщение без цвета):
print_info()    { echo -e "${CYAN}ℹ️  INFO:${NC} $1"; }
print_success() { echo -e "${GREEN}✅ SUCCESS:${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠️  WARNING:${NC} $1"; }
print_error()   { echo -e "${RED}❌ ERROR:${NC} $1"; }
print_step()    { echo -e "${CYAN}🔹 $1${NC}"; }
