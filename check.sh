#!/bin/bash

set -e

# Абсолютный путь к директории скрипта; подскрипты всегда берут конфиги
# отсюда, чтобы проверка работала одинаково из любой точки файловой системы:
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils.sh"

# Прямоугольник из псевдографики (╔═╗║╚═╝) с текстом по центру:
#   ╔═══════════════════════ Текст ═══════════════════════╗
#   ║                                                     ║
#   ╚═════════════════════════════════════════════════════╝
print_box() {
    local text="$1"
    local color="${2:-$CYAN}"
    local width
    width=$(get_terminal_width)
    local inner=$((width - 3))
    local text_length=${#text}
    local half=$(( (inner - text_length) / 2 ))
    local lpad=$half
    local rpad=$(( inner - text_length - half ))
    local hline empty pad_h
    pad_h=$(printf "%$((inner))s" "")
    hline="${pad_h// /═}"
    empty=$(printf "%$((inner))s")

    echo
    echo -e "${color}╔${hline}╗${NC}"
    echo -e "${color}║${empty}║${NC}"
    echo -e "${color}║$(printf "%*s" "$lpad" "")${text}$(printf "%*s" "$rpad" "")║${NC}"
    echo -e "${color}║${empty}║${NC}"
    echo -e "${color}╚${hline}╝${NC}"
    echo
}

usage() {
    cat <<EOF
Использование: $(basename "$0") [-f|--fix] [-g|--git-only] [путь...]

Комбинированная проверка: сначала check-py.sh (Python), затем
check-infra.sh (Docker/shell/Markdown). Аргументы передаются обоим
скриптам одинаково.

Позиционные аргументы - проверяемые файлы или папки.
Без путей: из корня dl_utils — белый список py + текущая папка infra,
из любой другой папки — её содержимое.

-f, --fix       разрешить автофиксы (автофиксация в обоих скриптах);
                по умолчанию режим отчёта - файлы не изменяются
-g, --git-only  проверять только файлы, закоммиченные в git (удобно для CI)
EOF
}

# Разбор аргументов: пути - в цели, флаги - на месте:
FIX=0
GIT_ONLY=0
TARGETS=()
for arg in "$@"; do
    case $arg in
        -f | --fix) FIX=1 ;;
        -g | --git-only) GIT_ONLY=1 ;;
        -h | --help) usage; exit 0 ;;
        -*)
            echo "Неизвестный флаг: $arg" >&2
            usage >&2
            exit 2
            ;;
        *) TARGETS+=("$arg") ;;
    esac
done

# Сборка аргументов для подскриптов:
SCRIPT_ARGS=()
[ "$FIX" -eq 1 ] && SCRIPT_ARGS+=("-f")
[ "$GIT_ONLY" -eq 1 ] && SCRIPT_ARGS+=("-g")
SCRIPT_ARGS+=("${TARGETS[@]}")

print_box "▶ check-py.sh"
PY_FAILED=0
bash "$SCRIPT_DIR/check-py.sh" "${SCRIPT_ARGS[@]}" || PY_FAILED=1

print_box "▶ check-infra.sh"
INFRA_FAILED=0
bash "$SCRIPT_DIR/check-infra.sh" "${SCRIPT_ARGS[@]}" || INFRA_FAILED=1

if [ "$PY_FAILED" -eq 1 ] || [ "$INFRA_FAILED" -eq 1 ]; then
    print_box "✗ ИТОГ: ошибки в одном или обоих скриптах" "$RED"
    exit 1
fi

print_box "✓ ИТОГ: все проверки пройдены успешно" "$GREEN"
