#!/bin/bash
set -e

# Динамическая подстройка UID/GID внутри контейнера под хостового пользователя:
if [ -n "$LOCAL_UID" ] && [ "$LOCAL_UID" != "0" ] && [ -z "${_USER_INIT_DONE:-}" ]; then
    export _USER_INIT_DONE=1
    set +e
    # shellcheck disable=SC2153
    sed -i "s/^user:[^:]*:[^:]*:[^:]*:/user:x:$LOCAL_UID:$LOCAL_GID:/" /etc/passwd
    # shellcheck disable=SC2153
    sed -i "s/^user:[^:]*:[^:]*:/user:x:$LOCAL_GID:/" /etc/group
    # Меняем владельца только тех файлов, куда нужна запись:
    chown "$LOCAL_UID:$LOCAL_GID" /home/user
    chown -R "$LOCAL_UID:$LOCAL_GID" /home/user/.jupyter /home/user/.config /home/user/.cache /home/user/.local /home/user/.npm /home/user/.gitconfig /home/user/.selected_editor /home/user/.bashrc /home/user/.profile /home/user/.zshrc /home/user/.oh-my-zsh 2>/dev/null || true
    set -e
    exec gosu user bash "$0" "$@"
fi

log() { echo "[cmd.sh $(date '+%H:%M:%S')] $*"; }

# Пути к лог-файлам:
JUPYTER_LOG="/tmp/jupyter.log"
OPENCODE_LOG="/tmp/opencode.log"

cleanup() {
    log "Завершение работы..."
    # shellcheck disable=SC2046
    kill $(jobs -p) 2>/dev/null
    # Ждём корректного завершения фоновых сервисов до 10 с (как docker stop),
    # затем принудительно добиваем и выходим, чтобы PID1 не завис:
    local i
    for ((i = 0; i < 10; i++)); do
        # shellcheck disable=SC2015
        jobs -p | grep -q . || break
        sleep 1
    done
    # shellcheck disable=SC2046
    kill -KILL $(jobs -p) 2>/dev/null
    wait 2>/dev/null
    exit 0
}
trap cleanup SIGTERM SIGINT SIGQUIT

check_port() {
    local port=$1 name=$2
    if python -c "import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('0.0.0.0', $port)); s.close()" 2>/dev/null; then
        return 0
    fi
    log "ОШИБКА: $name: порт $port занят!" >&2
    exit 1
}

# ==== Отчёт о состоянии =========================================================
# Рисуем здесь и запускаем сервисы, и отчёт о них: отчёт уходит и в лог контейнера,
# и в интерактивный баннер. Баннер лишь запрашивает готовые строки по каналу
# "пара FIFO" и печатает их дословно — своей отрисовки у него не остаётся.

# Цвета задаём именно escape-байтами ($'...'): иначе printf не развернёт их
# в строках-аргументах и в баннер уедут литералы "\033[0;90m".
CYAN=$'\033[0;36m'
YELLOW=$'\033[1;33m'
GRAY=$'\033[0;90m'
GREEN=$'\033[0;32m'
NC=$'\033[0m'

# Каталог канала "запрос-ответ":
SVC_DIR="${DL_CV_SVC_DIR:-/tmp/dl_cv}"
SVC_REQ="$SVC_DIR/req"
SVC_RES="$SVC_DIR/res"
# Ширина таблицы без рамки; заполняется на время отрисовки:
SVC_INNER=0

# Зарегистрированные сервисы: id → имя, внешний порт, pid.
# Pid сохраняется и после падения процесса: так состояние "упал" отличается
# от "не запущен" (pid = 0, сервис не запускался вовсе).
declare -A SVC_NAME SVC_PORT SVC_PID
SVC_ORDER=()

# Регистрация сервиса в отчёте: id, имя, внешний порт, pid (0 — не запущен).
svc_add() {
    SVC_NAME["$1"]="$2"
    SVC_PORT["$1"]="$3"
    SVC_PID["$1"]="$4"
    SVC_ORDER+=("$1")
}

# Отвечает ли кто-то на указанном порту (запасной признак запущенного сервиса):
svc_port_alive() {
    python -c "import socket, sys; s = socket.socket(); s.settimeout(1); sys.exit(s.connect_ex(('127.0.0.1', $1)))" 2>/dev/null
}

# Состояние сервиса на момент запроса: on, dead или off:
svc_state() {
    local id=$1 pid=${SVC_PID[$id]:-0} port=${SVC_PORT[$id]:-0}
    if [ "$pid" -gt 0 ]; then
        if kill -0 "$pid" 2>/dev/null; then echo on; else echo dead; fi
    elif [ "$port" -gt 0 ] && svc_port_alive "$port"; then
        # Pid неизвестен: сервис могли поднять вручную, судим по порту.
        echo on
    else
        echo off
    fi
}

# Реальные адреса сетей: перебираем интерфейсы и отбрасываем виртуальные
# (docker, veth, мосты), чтобы в отчёте был адрес, доступный снаружи контейнера:
svc_lan_ips() {
    python -c '
import fcntl, socket, struct
SKIP = ("lo", "docker", "br-", "veth", "cni", "flannel", "virbr", "tun", "tap")
addrs = []
for _, name in socket.if_nameindex():
    if name == "lo" or name.startswith(SKIP):
        continue
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        raw = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", name.encode()[:15]))
        addrs.append(socket.inet_ntoa(raw[20:24]))
    except OSError:
        pass  # у интерфейса может не быть адреса IPv4
print(" ".join(addrs))
' 2>/dev/null
}

# Строка таблицы: $1 — текст с управляющими последовательностями, $2 — длина
# текста без них (выравнивание считаем по ней), $3 — 1, если строка в рамке.
svc_row() {
    if [ "${3:-0}" -eq 1 ]; then
        local pad=$(( SVC_INNER - 2 - $2 ))
        [ "$pad" -lt 0 ] && pad=0
        # shellcheck disable=SC2059
        printf "${CYAN}║${NC} %s%*s ${CYAN}║${NC}\n" "$1" "$pad" ''
    else
        # Вне рамки хвостовые пробелы не нужны:
        printf "%s\n" "${1%"${1##*[![:space:]]}"}"
    fi
}

# Разделитель секций: вертикали внешней границы двойные, ответвление
# горизонтальное одинарное — глифы ╟ и ╢.
svc_sep() {
    # shellcheck disable=SC2059
    printf "${CYAN}╟%s╢${NC}\n" "$1"
}

# Секция контактов: адреса сетей и сервисы с портами и состоянием.
# $1 — ширина таблицы, $2 — 1, если печатаем внутри рамки:
svc_contacts() {
    local width=$1 framed=${2:-0} i
    [ "$width" -ge 40 ] 2>/dev/null || width=80
    SVC_INNER=$(( width - 2 ))
    # Ширины колонок: имя (обрезается по месту) и порт, состояние — маркером.
    # Заголовка колонок нет: порт отмечен двоеточием в каждой строке.
    local wn=12 wp=4 fill id st mark
    for id in "${SVC_ORDER[@]}"; do
        [ "${#SVC_NAME[$id]}" -gt "$wn" ] && wn=${#SVC_NAME[$id]}
    done
    [ "$wn" -gt $(( SVC_INNER - wp - 12 )) ] && wn=$(( SVC_INNER - wp - 12 ))
    [ "$wn" -lt 4 ] && wn=4
    fill=$(( SVC_INNER - wn - wp - 8 ))
    [ "$fill" -ge 0 ] || fill=0

    # Адреса сетей — чтобы было видно, откуда сервисы доступны снаружи:
    local ips plain
    ips="$(svc_lan_ips || true)"
    if [ -n "$ips" ]; then
        plain="СЕТЬ  $ips"
        svc_row "${YELLOW}СЕТЬ${NC}  ${YELLOW}${ips}${NC}" "${#plain}" "$framed"
    fi

    for id in "${SVC_ORDER[@]}"; do
        st="$(svc_state "$id")"
        case $st in
            on) mark="${GREEN}●${NC}" ;;
            dead) mark="${YELLOW}●${NC}" ;;
            *) mark="${GRAY}○${NC}" ;;
        esac
        svc_row "$mark $(printf "%-${wn}s  " "${SVC_NAME[$id]}")${GRAY}:${NC} ${YELLOW}$(printf "%-${wp}s" "${SVC_PORT[$id]}")${NC}$(printf "%${fill}s" '')" \
            "$(( 2 + wn + 2 + 2 + wp + fill ))" "$framed"
    done
}

# Вся таблица: рамка, система, GPU и контакты сервисов. $1 — ширина в колонках.
svc_banner() {
    # Паддинг по кириллице корректен только в многобайтовой локали:
    local LC_ALL=C.UTF-8
    local width=${1:-80} i inner top mid
    [ "$width" -ge 40 ] 2>/dev/null || width=80
    inner=$(( width - 2 ))
    SVC_INNER=$inner
    top=""
    mid=""
    for ((i = 0; i < inner; i++)); do
        top+="═"
        mid+="─"
    done
    # shellcheck disable=SC2059
    printf "${CYAN}╔%s╗${NC}\n" "$top"
    local txt
    txt="dl_cv  |  $(hostname)  |  $(date '+%d.%m.%Y %H:%M')"
    svc_row "$txt" "${#txt}" 1
    svc_sep "$mid"

    svc_contacts "$width" 1
    svc_sep "$mid"

    local cores
    cores="$(nproc 2>/dev/null || echo n/a)"
    txt="RAM $(free -h 2>/dev/null | awk '/Mem:/ {print $3 "/" $2}')   CPU $cores cores   Uptime: $(uptime -p 2>/dev/null | sed 's/up //')"
    svc_row "$txt" "${#txt}" 1
    local cuda
    cuda="$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | awk '{print $2}')"
    [ -z "$cuda" ] && cuda="${CUDA_VERSION:-n/a}"
    txt="$(python3 --version 2>/dev/null || echo "Python n/a")   CUDA $cuda"
    svc_row "$txt" "${#txt}" 1
    svc_sep "$mid"

    # GPU (если nvidia-smi доступен и отвечает):
    if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
        while IFS= read -r txt; do
            svc_row "${GRAY}GPU ${txt}${NC}" "$(( 4 + ${#txt} ))" 1
        done < <(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null)
    else
        txt="GPU недоступен (nvidia-smi)"
        svc_row "${YELLOW}${txt}${NC}" "${#txt}" 1
    fi

    # shellcheck disable=SC2059
    printf "${CYAN}╚%s╝${NC}\n" "$top"
}

# Канал "запрос-ответ" с баннером: ждём строку запроса, отдаём готовые строки.
# Ответ уходит через tee с таймаутом — если клиент исчез, открытие FIFO
# на запись заблокировало бы сервер навсегда:
svc_serve() {
    local req answer
    while read -r req < "$SVC_REQ"; do
        case $req in
            banner\ *) answer="$(svc_banner "${req#banner }" || echo "не удалось построить таблицу")" ;;
            services\ *) answer="$(svc_contacts "${req#services }" 0 || echo "не удалось построить список сервисов")" ;;
            *) answer="неизвестный запрос: $req" ;;
        esac
        printf '%s\n' "$answer" | timeout 5 tee "$SVC_RES" > /dev/null || :
    done
}

log "Запуск ollm_utils.py..."
T0=$SECONDS
python /workspace/*/dl_utils/ollm_utils.py \
    || log "ollm_utils: конфигурация моделей пропущена (нет Ollama?)."
log "ollm_utils.py выполнен за $((SECONDS - T0))с"

log "Проверка порта Jupyter..."
check_port "${JUPYTER_PORT:-8888}" "Jupyter Lab"
JP_CMD=(jupyter-lab --notebook-dir='/workspace' --ip='*' --no-browser --allow-root)
JP_CMD+=(--port="${JUPYTER_PORT:-8888}" --ServerApp.port_retries=0)
if [ -n "$JUPYTER_PASS" ]; then
    JP_HASH=$(echo "$JUPYTER_PASS" \
        | python -c 'from jupyter_server.auth import passwd; print(passwd(input()))')
    JP_CMD+=(--PasswordIdentityProvider.hashed_password="$JP_HASH")
fi

log "Запуск Jupyter Lab..."
"${JP_CMD[@]}" > "$JUPYTER_LOG" 2>&1 &
JPID=$!
svc_add jupyter "Jupyter Lab" "${JUPYTER_PORT:-8888}" "$JPID"
log "Jupyter Lab запущен (PID: $JPID) — порт ${JUPYTER_PORT:-8888} — лог: $JUPYTER_LOG"

# OpenCode v2 без заданного пароля сгенерирует случайный и напечатает его
# в лог, поэтому без OPENCODE_PASS сервер просто не запускаем.
export OPENCODE_SERVER_PASSWORD="${OPENCODE_PASS:-}"
OCID=""
OC_PORT="${OPENCODE_PORT:-8000}"

# OpenCode работает в сетевом пространстве хоста (см. run.sh), поэтому
# дефолтный порт его фонового сервиса вычисляется из идентификатора
# пользователя и конфликтует с таким же сервисом на хосте. Фиксируем
# порт в конфиге сервиса: тогда он подхватывается и при запуске CLI
# без параметров, и после перезапуска сервиса:
opencode service set port "${OPENCODE_SERVICE_PORT:-4097}" \
    || log "Не удалось задать порт фонового сервиса OpenCode."

if [ -n "$OPENCODE_SERVER_PASSWORD" ]; then
    log "Проверка порта OpenCode..."
    check_port "$OC_PORT" "OpenCode"
    cd /workspace
    OC_CMD=(opencode serve --hostname 0.0.0.0 --port="$OC_PORT")

    log "Запуск OpenCode..."
    "${OC_CMD[@]}" > "$OPENCODE_LOG" 2>&1 &
    OCID=$!
    svc_add opencode "OpenCode Web" "$OC_PORT" "$OCID"
    log "OpenCode запущен (PID: $OCID) — порт $OC_PORT — лог: $OPENCODE_LOG"
else
    svc_add opencode "OpenCode Web" "$OC_PORT" 0
    log "OPENCODE_PASS не задан, OpenCode не запускается."
fi

# Отчёт о состоянии печатается в лог контейнера и отдаётся баннеру по запросу:
svc_banner "${DL_CV_BANNER_WIDTH:-80}"

if mkdir -p "$SVC_DIR" && rm -f "$SVC_REQ" "$SVC_RES" && mkfifo -m 666 "$SVC_REQ" "$SVC_RES"; then
    svc_serve < /dev/null &
    log "Баннер запрашивает состояние сервисов через $SVC_REQ"
else
    log "⚠ Не удалось создать канал запросов в $SVC_DIR — баннер покажет заглушку." >&2
fi

log "Мониторинг..."

while true; do
    sleep 10
    JP_ALIVE=false; kill -0 "$JPID" 2>/dev/null && JP_ALIVE=true

    if ! $JP_ALIVE; then
        log "⚠ Jupyter Lab (PID $JPID) завершился! Лог:"
        tail -5 "$JUPYTER_LOG" 2>/dev/null | sed 's/^/  /'
        if [ -n "$OCID" ]; then
            log "(OpenCode продолжает работу)"
        fi
        JPID=""
    fi

    if [ -n "$OCID" ]; then
        OC_ALIVE=false; kill -0 "$OCID" 2>/dev/null && OC_ALIVE=true
        if ! $OC_ALIVE; then
            log "⚠ OpenCode (PID $OCID) завершился! Лог:"
            tail -5 "$OPENCODE_LOG" 2>/dev/null | sed 's/^/  /'
            log "(Jupyter Lab продолжает работу)"
            OCID=""
        fi
    fi

    if [ -z "$JPID" ] && [ -z "$OCID" ]; then
        log "Все процессы завершились. Выход."
        break
    fi

    if [ -z "$JPID" ] || [ -z "$OCID" ]; then
        log "--- alive check ---"
        if [ -n "$JPID" ]; then
            log "Jupyter (PID $JPID) жив"
        fi
        if [ -n "$OCID" ]; then
            log "OpenCode (PID $OCID) жив"
        fi
    fi
done

# shellcheck disable=SC2046
kill $(jobs -p) 2>/dev/null
wait 2>/dev/null
exit 1
