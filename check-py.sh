#!/bin/bash

# Файлы из корня репозитория, которые проверяются в дефолтном режиме запуска
# без аргументов из корня dl_utils; временный костыль - со временем от белых
# списков планируется отказаться совсем:
root_files=("labels.py" "pt_utils.py" "onnx_utils.py" "ollm_utils.py" "boxmot_utils.py" "ul_utils.py" "sam3al.py" "docker/set_symbolic_flag.py")

set -e  # Выход при первой ошибке

# Абсолютный путь к директории скрипта: конфиги линтеров всегда берутся отсюда,
# чтобы проверка работала одинаково из любой точки файловой системы:
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUFF_CONFIG="$SCRIPT_DIR/pyproject.toml"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils.sh"

usage() {
    cat <<EOF
Использование: $(basename "$0") [-f|--fix] [-g|--git-only] [путь...]

Позиционные аргументы - проверяемые файлы или папки (.py, .ipynb).
Без путей: запуск из корня dl_utils проверяет белый список,
из любой другой папки - её содержимое.

-f, --fix       разрешить автофиксы (ruff format и ruff check --unsafe-fixes);
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

# Список проваленных этапов; наполняется по ходу проверок:
FAILED_STAGES=()

# Фиксация проваленного этапа с продолжением остальных проверок:
mark_failure() {
    FAILED_STAGES+=("$1")
    print_error "Этап провален: $1"
}

# Подавление единственного допустимого варнинга ruff — о конфликте правила
# COM812 (trailing comma) с форматтером; остальные предупреждения проходят:
suppress_com812_warning() {
    grep -v 'may cause conflicts when used with the formatter' >&2
}

# Каталоги, исключаемые из поиска файлов на диске:
PRUNE_DIRS=(-name .git -o -name __pycache__ -o -name .venv -o -name venv \
    -o -name node_modules -o -name dist -o -name build \
    -o -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache)

# Режим работы: явные пути, либо дефолтное поведение по месту запуска:
if [ ${#TARGETS[@]} -eq 0 ]; then
    if [[ "$PWD" -ef "$SCRIPT_DIR" ]]; then
        MODE='legacy'
    else
        MODE='targets'
        TARGETS+=("$PWD")
    fi
else
    MODE='targets'
fi

# Каждая цель проверяется из собственной директории с путями относительно неё:
# послабления конфига (например, '**/tests/**' для assert в тестах) матчатся
# у любых внешних проектов, а mypy видит их собственные импорты:
TARGET_DIRS=()
declare -A MAIN_OF TEST_OF SEEN_ROOT SEEN_FILE

# Регистрация корня цели без дублей:
add_root() {
    local root=$1
    if [[ -z ${SEEN_ROOT[$root]:-} ]]; then
        SEEN_ROOT[$root]=1
        TARGET_DIRS+=("$root")
        MAIN_OF[$root]=''
        TEST_OF[$root]=''
    fi
}

# Добавление файла в бакет цели без дублей:
add_file() {
    local root=$1 rel=$2 kind=$3 key="$1/$2"
    if [[ -z ${SEEN_FILE[$key]:-} ]]; then
        SEEN_FILE[$key]=1
        if [[ $kind == test ]]; then
            TEST_OF[$root]+="${TEST_OF[$root]:+$'\n'}$rel"
        else
            MAIN_OF[$root]+="${MAIN_OF[$root]:+$'\n'}$rel"
        fi
    fi
}

is_test_file() {
    case $1 in
        test_*.py | *_test.py) return 0 ;;
        *) return 1 ;;
    esac
}

# Тройка линтеров для одного файла (.py или .ipynb); поведение ruff зависит от
# флага --fix; для .ipynb mypy вызывается через nbqa; путь передаётся
# относительно текущего каталога (корня цели):
check_one_file() {
    local display=$1
    local file=$2

    # Ruff format:
    print_separator "Ruff format: $display" "$CYAN"
    if [ "$FIX" -eq 1 ]; then
        if ruff format --config "$RUFF_CONFIG" "$file" 2> >(suppress_com812_warning); then
            print_success "Форматирование завершено"
        else
            mark_failure "ruff format: $display"
        fi
    else
        if ruff format --check --diff --config "$RUFF_CONFIG" "$file" 2> >(suppress_com812_warning); then
            print_success "Формат в порядке"
        else
            mark_failure "ruff format: $display"
        fi
    fi

    # Ruff check:
    print_separator "Ruff check: $display" "$CYAN"
    local -a check_args=(check --config "$RUFF_CONFIG")
    if [ "$FIX" -eq 1 ]; then
        check_args+=(--fix --unsafe-fixes)
    fi
    if ruff "${check_args[@]}" "$file" 2> >(suppress_com812_warning); then
        print_success "Проверка завершена"
    else
        mark_failure "ruff check: $display"
    fi

    # Mypy: для .ipynb используется обёртка nbqa, т.к. mypy не понимает
    # формат notebook нативно:
    print_separator "Mypy: $display" "$PURPLE"
    local -a mypy_cmd=(mypy --config-file "$RUFF_CONFIG")
    if [[ $file == *.ipynb ]]; then
        mypy_cmd=(nbqa mypy)
    fi
    if "${mypy_cmd[@]}" "$file"; then
        print_success "Типы в порядке"
    else
        mark_failure "mypy: $display"
    fi
}

# Прогон тройки линтеров по всем целям; второй аргумент - имя ассоциативного
# массива "корень -> список относительных путей":
run_stage() {
    local label=$1
    local -n bucket=$2
    print_separator "$label" "$CYAN"

    local total=0 root rel lines
    for root in "${TARGET_DIRS[@]}"; do
        [[ -n ${bucket[$root]} ]] || continue
        lines="$(grep -c . <<<"${bucket[$root]}")"
        total=$((total + lines))
    done
    if [ "$total" -eq 0 ]; then
        print_warning "Подходящих файлов не найдено"
        return 0
    fi

    for root in "${TARGET_DIRS[@]}"; do
        [[ -n ${bucket[$root]} ]] || continue
        pushd "$root" >/dev/null
        while IFS= read -r rel; do
            [[ -n $rel ]] || continue
            check_one_file "$root/$rel" "$rel"
        done <<<"${bucket[$root]}"
        popd >/dev/null
    done
    print_success "Этап завершён ($total файлов)"
}

# Сбор py и ipynb файлов целей в структуры "корень -> относительные пути":
# только закоммиченные при --git-only, иначе всё найденное на диске
# за вычетом служебных каталогов:
collect_targets() {
    local t abs root rel p prefix groot
    local -a spec
    for t in "${TARGETS[@]}"; do
        if [[ ! -e $t ]]; then
            print_error "Путь не найден: $t"
            exit 1
        fi
        if [[ -d $t ]]; then
            abs="$(cd "$t" && pwd)"
            root=$abs
        else
            abs="$(cd "$(dirname "$t")" && pwd)/$(basename "$t")"
            root=${abs%/*}
        fi
        add_root "$root"

        if [ "$GIT_ONLY" -eq 1 ]; then
            groot="$(git -C "$abs" rev-parse --show-toplevel)" || {
                print_error "--git-only требует git-репозиторий, а цель вне его: $abs"
                exit 1
            }
            # git ls-files отдаёт пути от корня репозитория - обрезаем префикс,
            # чтобы получить путь относительно цели; сам файл цели приходит
            # без префикса только если он закоммичен:
            prefix="$(realpath --relative-to="$groot" "$abs")"
            if [[ -f $abs ]]; then
                spec=("$prefix")
            else
                spec=("${prefix%/}/*.py" "${prefix%/}/*.ipynb")
            fi
            while IFS= read -r p; do
                [ -n "$p" ] || continue
                rel=${p#"$prefix"}
                rel=${rel#/}
                [[ -n $rel ]] || rel=${prefix##*/}
                if is_test_file "${rel##*/}"; then
                    add_file "$root" "$rel" test
                else
                    add_file "$root" "$rel" main
                fi
            done < <(git -C "$groot" ls-files -c -- "${spec[@]}" 2>/dev/null || true)
        else
            if ! git -C "$abs" rev-parse --git-dir >/dev/null 2>&1; then
                print_warning "Вне git-репозитория - берутся все файлы с диска: $abs"
            fi
            # Одиночный файл на диске проверяется всегда:
            if [[ -f $abs ]]; then
                rel=${abs##*/}
                if is_test_file "$rel"; then
                    add_file "$root" "$rel" test
                else
                    add_file "$root" "$rel" main
                fi
                continue
            fi
            while IFS= read -r p; do
                [ -n "$p" ] || continue
                rel=${p#"$root"/}
                if is_test_file "${rel##*/}"; then
                    add_file "$root" "$rel" test
                else
                    add_file "$root" "$rel" main
                fi
            done < <(
                find "$abs" \( "${PRUNE_DIRS[@]}" \) -prune \
                    -o -type f \( -name '*.py' -o -name '*.ipynb' \) -print \
                    2>/dev/null || true
            )
        fi
    done

    # Детерминированный порядок файлов внутри каждой цели:
    local root
    local -a sorted
    for root in "${TARGET_DIRS[@]}"; do
        mapfile -t sorted < <(grep . <<<"${MAIN_OF[$root]}" | LC_ALL=C sort -u)
        ((${#sorted[@]} > 0)) && MAIN_OF[$root]="$(printf '%s\n' "${sorted[@]}")"
        mapfile -t sorted < <(grep . <<<"${TEST_OF[$root]}" | LC_ALL=C sort -u)
        ((${#sorted[@]} > 0)) && TEST_OF[$root]="$(printf '%s\n' "${sorted[@]}")"
    done
    return 0
}

echo -e "${GREEN}🚀 Запуск проверок качества кода и тестов...${NC}"

# Версии инструментов в шапке: дрейф версий сразу виден при странных прогонах:
RUFF_V="$(ruff --version 2>/dev/null || true)"
MYPY_V="$(mypy --version 2>/dev/null || true)"
NBQA_V="$(nbqa --version 2>/dev/null || true)"
print_step "Инструменты: ${RUFF_V:-нет ruff}, ${MYPY_V:-нет mypy}, ${NBQA_V:-нет nbqa}"
if [ "$FIX" -eq 1 ]; then
    print_step "Режим правки (-f): автофиксы разрешены"
else
    print_step "Режим отчёта: файлы не изменяются (автофиксы - ключ -f)"
fi

if [ "$MODE" = 'legacy' ]; then
    print_step "Дефолтный запуск из корня dl_utils - белый список"

    # Проверяем Git-репозиторий относительно директории скрипта:
    if ! git -C "$SCRIPT_DIR" rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Этот скрипт должен запускаться внутри Git-репозитория"
        exit 1
    fi
    GIT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

    add_root "$GIT_ROOT"
    for f in "${root_files[@]}"; do
        add_file "$GIT_ROOT" "$f" main
    done
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        add_file "$GIT_ROOT" "$f" test
    done < <(git -C "$GIT_ROOT" ls-files -c -o --exclude-standard -- 'tests/*.py' 2>/dev/null || true)
else
    print_step "Целевые пути: ${TARGETS[*]}"
    collect_targets
fi

run_stage "Проверка основных файлов" MAIN_OF

# pytest запускается отдельно в каждой цели со своими тестами - каждый проект
# тестируется в собственном контексте (его rootdir, pythonpath, строгость):
ANY_TESTS=0
for root in "${TARGET_DIRS[@]}"; do
    [[ -n ${TEST_OF[$root]} ]] || continue
    ANY_TESTS=1
done

if [ "$ANY_TESTS" -eq 1 ]; then
    print_separator "Запуск тестов" "$YELLOW"
    for root in "${TARGET_DIRS[@]}"; do
        [[ -n ${TEST_OF[$root]} ]] || continue
        mapfile -t test_list < <(grep . <<<"${TEST_OF[$root]}")
        if (cd "$root" && pytest -v "${test_list[@]}"); then
            print_success "Тесты прошли: $root"
        else
            mark_failure "pytest: $root"
        fi
    done

    run_stage "Проверка тестов" TEST_OF
else
    print_info "Тесты не найдены - pytest пропущен"
fi

# Финальный вердикт: скрипт успешен только при отсутствии проваленных этапов:
print_separator "ИТОГ" "$CYAN"
if [ ${#FAILED_STAGES[@]} -gt 0 ]; then
    print_error "Проваленных этапов: ${#FAILED_STAGES[@]}"
    for stage in "${FAILED_STAGES[@]}"; do
        print_error "  - $stage"
    done
    exit 1
fi

print_separator "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!" "$GREEN"
echo -e "${GREEN}🎉🎉🎉 Поздравляем! Все проверки завершены успешно! 🎉🎉🎉${NC}"
