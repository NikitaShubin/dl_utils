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

# Код возврата при отсутствии файлов подходящего типа (.py, .ipynb):
# возвращается только в режиме тишины (-q); в CI это честный nonzero, чтобы
# покровная проверка не давала ложный «зелёный» по пустому списку файлов:
NO_FILES=3

usage() {
    cat <<EOF
Использование: $(basename "$0") [-f|--fix] [-g|--git-only] [-q] [-H] [путь...]

Позиционные аргументы - проверяемые файлы или папки (.py, .ipynb).
Без путей: запуск из корня dl_utils проверяет белый список,
из любой другой папки - её содержимое.

-f, --fix       разрешить автофиксы (ruff format и ruff check --unsafe-fixes);
                по умолчанию режим отчёта - файлы не изменяются
-g, --git-only  проверять только файлы, закоммиченные в git (удобно для CI)
-q, --quiet-no-files  в режиме тишины (для главного check.sh): при отсутствии
                файлов подходящего типа ничего не печатать и выйти с кодом 3;
                иначе вывести сообщение об отсутствии и выйти с кодом 0
-H, --print-header  печатать шапку (заголовок, версии, цели) - для check.sh;
                без флага шапка подавлена
EOF
}

# Разбор аргументов: пути - в цели, флаги - на месте:
FIX=0
GIT_ONLY=0
QUIET=0
PRINT_HEADER=0
TARGETS=()
for arg in "$@"; do
    case $arg in
        -f | --fix) FIX=1 ;;
        -g | --git-only) GIT_ONLY=1 ;;
        -q | --quiet-no-files) QUIET=1 ;;
        -H | --print-header) PRINT_HEADER=1 ;;
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

# Флаги окраски для внешних инструментов (вычисляются здесь, где stderr -
# настоящий терминал, а не при захвате вывода в run_linter):
COLOR_RUFF="$(color_flags ruff)"
COLOR_MYPY="$(color_flags mypy)"
COLOR_PYTEST="$(color_flags pytest)"

# Фиксация проваленного этапа с продолжением остальных проверок:
mark_failure() {
    FAILED_STAGES+=("$1")
    print_error "Этап провален: $1"
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

# Поднимает корень до родителя ближайшего каталога tests, если цель лежит в нём
# или глубже: в относительных путях должен оставаться компонент tests/, иначе
# per-file-ignores ruff вида '**/tests/**' (S101, PLR2004 в тестах) не
# срабатывают при явном указании файла или самого каталога tests/:
lift_tests_root() {
    local d=$1
    while [[ -n ${d%/*} && ${d##*/} != tests ]]; do
        d=${d%/*}
    done
    if [[ -n ${d%/*} ]]; then
        echo "${d%/*}"
    else
        echo "$1"
    fi
}

# Прогон одного линтера: вывод захватывается и печатается только при неудаче,
# чтобы в норме не было шума; имя линтера уходит в маркер провала этапа:
run_linter() {
    local name=$1 display=$2
    shift 2
    local out
    if out=$("$@" 2>&1); then
        return 0
    fi
    out="$(printf '%s\n' "$out" | grep -v 'may cause conflicts when used with the formatter')"
    [[ -n $out ]] && printf '%s\n' "$out"
    mark_failure "$name: $display"
}

# Тройка линтеров для одного файла (.py или .ipynb); поведение ruff зависит от
# флага --fix; для .ipynb mypy вызывается через nbqa; поток в каждой цели
# идёт с путями относительно её корня:
check_one_file() {
    local display=$1
    local file=$2
    echo -e "${CYAN}▸ ${MAGENTA}${display}${NC}"

    local -a check_args=(check --config "$RUFF_CONFIG" "$COLOR_RUFF")
    if [ "$FIX" -eq 1 ]; then
        check_args+=(--fix --unsafe-fixes)
    fi

    if [ "$FIX" -eq 1 ]; then
        run_linter "ruff format" "$display" ruff format --config "$RUFF_CONFIG" "$COLOR_RUFF" "$file"
    else
        run_linter "ruff format" "$display" ruff format --check --diff --config "$RUFF_CONFIG" "$COLOR_RUFF" "$file"
    fi
    run_linter "ruff check" "$display" ruff "${check_args[@]}" "$file"

    # Mypy: для .ipynb используется обёртка nbqa, т.к. mypy не понимает
    # формат notebook нативно:
    local -a mypy_cmd=(mypy --config-file "$RUFF_CONFIG" "$COLOR_MYPY")
    if [[ $file == *.ipynb ]]; then
        mypy_cmd=(nbqa mypy)
    fi
    run_linter "mypy" "$display" "${mypy_cmd[@]}" "$file"
}

# Прогон тройки линтеров по всем целям; второй аргумент - имя ассоциативного
# массива "корень -> список относительных путей":
run_stage() {
    local label=$1
    local -n bucket=$2
    print_separator "$label" "$CYAN"

    local total=0 failed_files=0 failed_before root rel lines
    for root in "${TARGET_DIRS[@]}"; do
        [[ -n ${bucket[$root]} ]] || continue
        lines="$(grep -c . <<<"${bucket[$root]}")"
        total=$((total + lines))
    done
    if [ "$total" -eq 0 ]; then
        print_info "Файлы не найдены"
        return 0
    fi

    for root in "${TARGET_DIRS[@]}"; do
        [[ -n ${bucket[$root]} ]] || continue
        pushd "$root" >/dev/null
        while IFS= read -r rel; do
            [[ -n $rel ]] || continue
            failed_before=${#FAILED_STAGES[@]}
            check_one_file "$root/$rel" "$rel"
            [[ ${#FAILED_STAGES[@]} -gt failed_before ]] && failed_files=$((failed_files + 1))
        done <<<"${bucket[$root]}"
        popd >/dev/null
    done

    if [ "$failed_files" -eq 0 ]; then
        print_success "Этап завершён ($total файлов)"
    else
        print_error "Этап завершён с ошибками ($failed_files из $total файлов)"
    fi
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
        root="$(lift_tests_root "$root")"
        add_root "$root"

        # git работает с каталогами: для целевого файла берём его родителя:
        if [[ -d $abs ]]; then
            git_ctx=$abs
        else
            git_ctx=${abs%/*}
        fi
        if [ "$GIT_ONLY" -eq 1 ]; then
            groot="$(git -C "$git_ctx" rev-parse --show-toplevel)" || {
                print_error "--git-only требует git-репозиторий, а цель вне его: $abs"
                exit 1
            }
            # git ls-files отдаёт пути от корня репозитория - обрезаем префикс,
            # чтобы получить путь относительно корня цели (возможно поднятого
            # над тестами); сам файл цели приходит без префикса только если
            # он закоммичен:
            prefix="$(realpath --relative-to="$groot" "$abs")"
            root_grel="$(realpath --relative-to="$groot" "$root")"
            [[ $root_grel == '.' ]] && root_grel=''
            if [[ -f $abs ]]; then
                case $abs in
                    *.py | *.ipynb) spec=("$prefix") ;;
                    *) continue ;;
                esac
            else
                spec=("${prefix%/}/*.py" "${prefix%/}/*.ipynb")
            fi
            while IFS= read -r p; do
                [ -n "$p" ] || continue
                rel=${p#"$root_grel"/}
                [[ -n $rel ]] || rel=${abs#"$root"/}
                if is_test_file "${rel##*/}"; then
                    add_file "$root" "$rel" test
                else
                    add_file "$root" "$rel" main
                fi
            done < <(git -C "$groot" ls-files -c -- "${spec[@]}" 2>/dev/null || true)
        else
            if ! git -C "$git_ctx" rev-parse --git-dir >/dev/null 2>&1; then
                [ "$QUIET" -eq 1 ] || print_warning "Вне git-репозитория - берутся все файлы с диска: $abs"
            fi
            # Одиночный файл проверяется только если подходит по типу (.py/.ipynb):
            if [[ -f $abs ]]; then
                case $abs in
                    *.py | *.ipynb)
                        rel=${abs#"$root"/}
                        if is_test_file "${rel##*/}"; then
                            add_file "$root" "$rel" test
                        else
                            add_file "$root" "$rel" main
                        fi
                        ;;
                esac
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

# Сбор и классификация файлов целей по бакетам «основные» и «тесты»:
if [ "$MODE" = 'legacy' ]; then
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
    collect_targets
fi

# Подсчёт найденных файлов подходящего типа (.py, .ipynb):
FOUND=0
if [ "$MODE" = 'legacy' ] || [ ${#TARGET_DIRS[@]} -gt 0 ]; then
    for root in "${TARGET_DIRS[@]}"; do
        [[ -n ${MAIN_OF[$root]} || -n ${TEST_OF[$root]} ]] && FOUND=1
    done
fi

# Нет файлов подходящего типа: в режиме тишины (-q) ничего не печатаем и
# отдаём код 3 (для check.sh и честного CI), иначе - инфо-сообщение и код 0:
if [ "$FOUND" -eq 0 ]; then
    if [ "$QUIET" -eq 0 ]; then
        print_warning "Нет файлов подходящего типа (.py, .ipynb)"
    fi
    exit $((QUIET ? NO_FILES : 0))
fi

# Шапка: заголовок-box, версии инструментов, режим и цели; печатается только
# при флаге -H (его ставит главный check.sh) и раз файлы есть - уместна:
if [ "$PRINT_HEADER" -eq 1 ]; then
    print_box "▶ check-py.sh"
    print_step "Запуск проверок качества кода и тестов"

    # Версии инструментов: дрейф версий сразу виден при странных прогонах:
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
    else
        print_step "Целевые пути: ${TARGETS[*]}"
    fi
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
        # Вывод pytest захватывается и печатается только при неудаче (краткие
        # traceback и сводка), чтобы в норме не было шума из имён тестов:
        if out=$( (cd "$root" && pytest -q --tb=short "$COLOR_PYTEST" "${test_list[@]}" ) 2>&1); then
            print_success "Тесты прошли: $root"
        else
            [[ -n $out ]] && printf '%s\n' "$out"
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
print_success "Все проверки завершены успешно!"
