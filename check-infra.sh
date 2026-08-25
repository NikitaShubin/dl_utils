#!/bin/bash

set -e

# Абсолютный путь к директории скрипта: конфиги линтеров всегда берутся отсюда,
# чтобы проверка работала одинаково из любой точки файловой системы:
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils.sh"

usage() {
    cat <<EOF
Использование: $(basename "$0") [-f|--fix] [-g|--git-only] [путь...]

Позиционные аргументы - проверяемые файлы или папки
(Dockerfile, docker-compose, shell, markdown).
Без путей проверяется текущая директория.

-f, --fix       разрешить автофиксы (умеют dclint и markdownlint);
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

# Без путей целью становится текущая директория:
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS+=("$PWD")
fi

# Каталоги, исключаемые из поиска файлов на диске:
PRUNE_DIRS=(-name .git -o -name __pycache__ -o -name .venv -o -name venv \
    -o -name node_modules -o -name dist -o -name build \
    -o -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache)

# Нормализация целей в абсолютные пути:
ABS_TARGETS=()
for t in "${TARGETS[@]}"; do
    if [[ ! -e $t ]]; then
        print_error "Путь не существует: $t"
        exit 1
    fi
    if [[ -d $t ]]; then
        ABS_TARGETS+=("$(cd "$t" && pwd)")
    else
        ABS_TARGETS+=("$(cd "$(dirname "$t")" && pwd)/$(basename "$t")")
    fi
done

# Валидация репозиториев: при --git-only цель обязана быть в гите,
# иначе просто предупреждение и проверка всего с диска:
declare -A REPO_WARNED=()
for t in "${ABS_TARGETS[@]}"; do
    d="$t"
    if [[ -f $t ]]; then d="$(dirname "$t")"; fi
    if ! git -C "$d" rev-parse --git-dir >/dev/null 2>&1; then
        if [ "$GIT_ONLY" -eq 1 ]; then
            print_error "--git-only требует git-репозитория, а цель вне его: $d"
            exit 1
        fi
        if [ -z "${REPO_WARNED[$d]:-}" ]; then
            print_warning "Вне git-репозитория - берутся все файлы с диска: $d"
            REPO_WARNED[$d]=1
        fi
    fi
done

print_step "Целевые пути: ${ABS_TARGETS[*]}"
if [ "$FIX" -eq 1 ]; then
    print_info "Режим правки (-f): автофиксы разрешены"
else
    print_info "Режим отчёта: файлы не изменяются (автофиксы - ключ -f)"
fi
echo

# Кандидаты категории по цели: маски передаются аргументами; при --git-only
# это листинг git относительно корня репозитория, иначе поиск по диску:
emit_candidates() {
    local t=$1
    shift
    local root rel p s
    if [ "$GIT_ONLY" -eq 1 ]; then
        root="$(git -C "$t" rev-parse --show-toplevel 2>/dev/null)" || return 0
        rel="$(realpath --relative-to="$root" "$t")"
        if [ "$rel" = '.' ]; then rel=''; fi
        local -a spec=()
        for s in "$@"; do
            spec+=("${rel:+$rel/}$s")
        done
        while IFS= read -r p; do
            if [ -n "$p" ]; then printf '%s\n' "$root/$p"; fi
        done < <(git -C "$root" ls-files -c -- "${spec[@]}" 2>/dev/null || true)
        return 0
    fi
    local -a conds=()
    local first=1
    for s in "$@"; do
        if [ "$first" -eq 1 ]; then
            conds=(-name "$s")
            first=0
        else
            conds+=(-o -name "$s")
        fi
    done
    find "$t" \( "${PRUNE_DIRS[@]}" \) -prune \
        -o -type f \( "${conds[@]}" \) -print 2>/dev/null || true
    return 0
}

# Фильтры релевантности по типу файла:
is_dockerfile() {
    local b
    b="$(basename "$1")"
    [[ $b == Dockerfile || $b == *.Dockerfile ]]
}
is_compose() {
    local b
    b="$(basename "$1")"
    [[ $b == docker-compose*.yml || $b == docker-compose*.yaml || $b == compose*.yml || $b == compose*.yaml ]]
}
is_sh() {
    [[ $(basename "$1") == *.sh ]]
}
is_md() {
    [[ $(basename "$1") == *.md ]]
}

# Заполнение списка файлов категории целями через фильтр релевантности:
gather() {
    local out=$1 matcher=$2
    shift 2
    # Именная ссылка на массив по имени из аргумента; shellcheck не отслеживает
    # семантику nameref и считает присваивание строкой:
    # shellcheck disable=SC2178
    local -n files=$out
    files=()
    local t p
    for t in "${ABS_TARGETS[@]}"; do
        if [[ -f $t ]]; then
            if "$matcher" "$t"; then files+=("$t"); fi
        else
            while IFS= read -r p; do
                if [ -n "$p" ] && "$matcher" "$p"; then files+=("$p"); fi
            done < <(emit_candidates "$t" "$@")
        fi
    done
    if [ ${#files[@]} -gt 0 ]; then
        mapfile -t files < <(printf '%s\n' "${files[@]}" | LC_ALL=C sort -u)
    fi
    return 0
}

# Прогон одной категории: каждый файл проверяется из своей директории,
# неудача не останавливает остальные; счётчик проблемных файлов общий:
run_check() {
    local desc=$1 arr=$2 cfg=$3
    shift 3
    local -a cmd=("$@")
    # Именная ссылка на массив по имени из аргумента; shellcheck не отслеживает
    # семантику nameref и считает присваивание строкой:
    # shellcheck disable=SC2178
    local -n files=$arr
    local failed_before=$TOTAL_FAILED

    print_separator "Проверка $desc"

    if [ -n "$cfg" ]; then
        print_info "Конфигурационный файл: $cfg"
        echo
    fi

    local count=0 file
    for file in "${files[@]}"; do
        count=$((count + 1))
        echo -e "${CYAN}▸ ${MAGENTA}${file#"$PWD"/}${NC}"

        if (cd "$(dirname "$file")" && "${cmd[@]}" "$(basename "$file")"); then
            :
        else
            print_error "Ошибка проверки: $file"
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
        fi
    done

    if [ "$count" -eq 0 ]; then
        print_info "Файлы не найдены"
    else
        local category_failed=$((TOTAL_FAILED - failed_before))
        if [ "$category_failed" -eq 0 ]; then
            print_success "Проверка завершена ($count файлов)"
        else
            print_error "Проверка завершена с ошибками ($category_failed из $count файлов)"
        fi
    fi
    return 0
}

# Общий счётчик проблемных файлов по всем категориям проверок:
TOTAL_FAILED=0

# Экстра-аргументы автофиксов для поддерживающих их инструментов:
DCLINT_EXTRA=()
MDLINT_EXTRA=()
if [ "$FIX" -eq 1 ]; then
    DCLINT_EXTRA+=(--fix)
    MDLINT_EXTRA+=(--fix)
fi

# Сбор файлов по категориям:
gather FILES_COMPOSE is_compose 'docker-compose*.yml' 'docker-compose*.yaml' 'compose*.yml' 'compose*.yaml'
gather FILES_DOCKER is_dockerfile 'Dockerfile' '*.Dockerfile'
gather FILES_SH is_sh '*.sh'
gather FILES_MD is_md '*.md'

# Проверка docker-compose файлов:
cfg="$SCRIPT_DIR/.dclintrc"
run_check "🐙 docker-compose файлы" FILES_COMPOSE "$cfg" dclint -c "$cfg" "${DCLINT_EXTRA[@]}"

# Проверка Dockerfile:
cfg="$SCRIPT_DIR/.hadolint.yaml"
run_check "🐋 Dockerfile" FILES_DOCKER "$cfg" hadolint --config "$cfg"

# Проверка shell-скриптов; у shellcheck нет опции конфига: он сам ищет
# .shellcheckrc вверх по дереву от файла, поэтому вне дерева dl_utils
# могут применяться правила самого проекта:
run_check "🐚 shell-скрипты" FILES_SH '' shellcheck

# Проверка Markdown файлов:
cfg="$SCRIPT_DIR/.markdownlint.yaml"
run_check "📖 Markdown файлы" FILES_MD "$cfg" markdownlint --config "$cfg" "${MDLINT_EXTRA[@]}"

# Финальный вердикт: скрипт успешен только при отсутствии ошибок:
print_separator "ВСЕ ПРОВЕРКИ ЗАВЕРШЕНЫ"
if [ "$TOTAL_FAILED" -gt 0 ]; then
    print_error "Всего проблемных файлов: $TOTAL_FAILED"
    exit 1
fi
print_success "Все проверки прошли успешно!"
