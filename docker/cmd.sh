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
    chown -R "$LOCAL_UID:$LOCAL_GID" /home/user/.jupyter /home/user/.config /home/user/.cache /home/user/.local /home/user/.npm /home/user/.gitconfig /home/user/.selected_editor /home/user/.bashrc /home/user/.profile 2>/dev/null || true
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
log "Jupyter Lab запущен (PID: $JPID) — порт ${JUPYTER_PORT:-8888} — лог: $JUPYTER_LOG"

export OPENCODE_SERVER_PASSWORD="${OPENCODE_PASS:-}"
export OPENCODE_SERVER_USERNAME="user"

log "Чистка telemetry из конфига OpenCode..."
OPENCODE_CONFIG="/home/user/.config/opencode/opencode.jsonc"
if [ -f "$OPENCODE_CONFIG" ]; then
    sed -i '/"telemetry"/,+2d' "$OPENCODE_CONFIG" 2>/dev/null || true
fi

log "Установка @ai-sdk/openai-compatible..."
npm install -g @ai-sdk/openai-compatible 2>/dev/null || true

OC_PORT="${OPENCODE_PORT:-8000}"
log "Проверка порта OpenCode..."
check_port "$OC_PORT" "OpenCode"
cd /workspace
OC_CMD=(opencode web --hostname 0.0.0.0 --port="$OC_PORT")

log "Запуск OpenCode..."
"${OC_CMD[@]}" > "$OPENCODE_LOG" 2>&1 &
OCID=$!
log "OpenCode запущен (PID: $OCID) — порт $OC_PORT — лог: $OPENCODE_LOG"

log "Оба сервиса запущены. Мониторинг..."
log "Jupyter Lab: http://localhost:${JUPYTER_PORT:-8888}"
log "OpenCode:    http://localhost:$OC_PORT"

while true; do
    sleep 10
    JP_ALIVE=false; kill -0 "$JPID" 2>/dev/null && JP_ALIVE=true
    OC_ALIVE=false; kill -0 "$OCID" 2>/dev/null && OC_ALIVE=true

    if ! $JP_ALIVE && ! $OC_ALIVE; then
        log "Оба процесса завершились. Выход."
        break
    fi

    if ! $JP_ALIVE; then
        log "⚠ Jupyter Lab (PID $JPID) завершился! Лог:"
        tail -5 "$JUPYTER_LOG" 2>/dev/null | sed 's/^/  /'
        log "(OpenCode продолжает работу)"
        JPID=""
    fi

    if ! $OC_ALIVE; then
        log "⚠ OpenCode (PID $OCID) завершился! Лог:"
        tail -5 "$OPENCODE_LOG" 2>/dev/null | sed 's/^/  /'
        log "(Jupyter Lab продолжает работу)"
        OCID=""
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
