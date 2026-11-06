#!/usr/bin/env bash

#############################
# Вход в консоль контейнера #
#############################

# Пользователь внутри контейнера: по умолчанию - пользователь хоста,
# с флагом --root - root (так openroot.sh переиспользует этот скрипт):
EXEC_USER="$(id -u):$(id -g)"
if [ "${1:-}" = "--root" ]; then
    EXEC_USER=root
    shift
fi

# Задаём имя контейнера или берём его из входных параметров:
if [ $# -eq 0 ]; then
    DOCKER_NAME="dl_cv_$USER"
else
    DOCKER_NAME="$1"
    shift
fi

# Контейнер запущен? Ошибки docker (нет контейнера или демона) не показываем:
container_running() {
    [ "$(docker inspect --format '{{.State.Running}}' "$DOCKER_NAME" 2>/dev/null)" = "true" ]
}

# Пока cmd.sh не закончил подготовку окружения (смена владельца файлов на
# пользователя хоста) и не поднял сервисы, вход даёт ошибки прав в консоли и
# баннер-заглушку. Поэтому ждём строку мониторинга в логе - это отметка, что
# контейнер готов к работе:
WAIT_SECS=90
wait_ready() {
    # Остановленный или отсутствующий контейнер готовым не станет - не ждём:
    if ! container_running; then
        echo "⚠ Контейнер $DOCKER_NAME не запущен - сначала запустите run.sh." >&2
        return 1
    fi

    local i=0
    while [ "$i" -lt "$WAIT_SECS" ]; do
        if docker logs "$DOCKER_NAME" 2>&1 | grep -q "Мониторинг"; then
            return 0
        fi
        # Контейнер мог упасть, пока мы ждали готовности:
        if ! container_running; then
            echo "⚠ Контейнер $DOCKER_NAME остановился до готовности - открывать нечего." >&2
            return 1
        fi
        [ "$i" -eq 0 ] && echo "Ждём готовности контейнера..."
        sleep 1
        i=$(( i + 1 ))
    done
    # Контейнер жив, но отметки нет - открываем консоль, как раньше:
    echo "⚠ Контейнер не сообщил о готовности за $WAIT_SECS с - открываем консоль." >&2
}
wait_ready || exit 1

docker exec -it -u "$EXEC_USER" "$DOCKER_NAME" zsh
