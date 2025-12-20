#!/usr/bin/env bash

#####################
# Запуск контейнера #
#####################

# Определяем положение текущего скрипта:
DOCKERFILE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Запускаемый докерфайл должен распологаться в той же дирректории.

# Цвета текста:
GREEN='\033[0;32m'
RED='\033[0;31m'
# YELLOW='\033[1;33m'
# CYAN='\033[0;36m'
NC='\033[0m'

# Задаём имя контейнера или берём его из входных параметров:
if [ $# -eq 0 ]; then
    DOCKER_NAME="dl_cv_$USER"
else
    DOCKER_NAME="$1"
    shift
fi

# Определяем имя образа:
IMAGE_NAME="kitbrain/dl_cv"

# Определяем имя хоста внутри докера:
DOCKER_HOSTNAME="${DOCKER_NAME}_$(hostname)"

# Определяем теги образа:
GITTAG=GIT-$(git rev-parse --short HEAD)
DATETAG=$(date +"%Y.%m.%d")

# Строка образа с тегами:
IMAGE_NAME_AND_TAGS=(
    -t "$IMAGE_NAME:$GITTAG"
    -t "$IMAGE_NAME:$DATETAG"
    -t "$IMAGE_NAME:latest"
)

# Пытаемся собирать образ каждый раз заново:
#if ! docker build --progress=plain "${IMAGE_NAME_AND_TAGS[@]}" "${DOCKERFILE_DIR}";
if ! docker build "${IMAGE_NAME_AND_TAGS[@]}" "${DOCKERFILE_DIR}";
then
    # Если образ собрать не удалось - берём готовый с DockerHub:
    printf "%bОбраз не собран!\n" "$RED"
    printf "Берётся версия из DockerHub!\n%b" "$NC"
    docker pull $IMAGE_NAME
else
    # Если образ успешно собран, отправляем ВСЕ теги в одном фоновом процессе:
    nohup bash -c "
        echo 'Начало отправки образов'
        docker push '$IMAGE_NAME:$GITTAG'
        docker push '$IMAGE_NAME:$DATETAG'
        docker push '$IMAGE_NAME:latest'
        echo 'Отправка завершена'
    " > /dev/null 2>&1 &
    printf "%bОбраз отправляется на Docker HUB (PID: %s)%b\n" "$GREEN" "$!" "$NC"
fi

# Включаем доступ к Nvidia, если установлен nvidia-smi:
# Стало:
if nvidia-smi >/dev/null 2>&1; then
    nvidia_args=('--runtime=nvidia' '--gpus' 'all')
else
    nvidia_args=()
fi

# Путь до домашней папки:
home=$(realpath ~)

# Параметры запуска контейнера:
RUNPARAMS=(
    # Фоновый режим:
    -d

    # Права внешнего пользователя:
    # -u $(id -u):$(id -g)
    # -e HOME=/home/user
    # Без явного указания на домашнюю папку контейнер не заработает у пользователей с id != 1000!

    # Проброс SSH агента:
    # -v "$SSH_AUTH_SOCK:/tmp/ssh_agent.sock" \
    # -e SSH_AUTH_SOCK=/tmp/ssh_agent.sock

    # Для работы Keras Tuner по сети:
    --ipc=host
    --network="host"
    --add-host "${DOCKER_HOSTNAME}":127.0.1.1

    # Удаляем контейнер после завершения:
    --rm
    # Не совместим с --restart unless-stopped.

    # Автоматический перезапуск вместе с демоном, если
    # к моменту перезапуска контейнер не был остановлен вручную:
    # --restart unless-stopped
    # Не совместим с --rm.

    # Включаем доступ к GPU, если возможно:
    "${nvidia_args[@]}"

    # Монтируем папку проекта в докер:
    -v "${DOCKERFILE_DIR}/../../../":/workspace

    # Путь до датасета подменяем локальной его копией:
    # -v "$home/projects/AP2.0/local_files/ds/":"/workspace/data/processed"
    -v "$home/projects/IQF/local_files/ds/":"/workspace/data/processed"
    #-v "/":"/outroot"
    -v "/":"/outroot":ro
    #-v "$home/.ssh/":"/home/user/.ssh":ro

    # Имя контейнера:
    --name "${DOCKER_NAME}"

    # Пароль и порт для Jupyter:
    -e JUPYTER_PASS
    -e JUPYTER_PORT
    # Адрес локального Ollama-сервера:
    -e OLLAMA_HOST
    # Берутся из внешних переменных окружения.
    # Рекомендуется добавить строки вроде
    # ```
    # JUPYTER_PASS="my_password"; export JUPYTER_PASS
    # ```
    # в файл ~/.profile, если используется bash,
    # или ~/.zshenv при использовании zsh.
    # https://unix.stackexchange.com/a/21600

    # Сетевое имя:
    -h "${DOCKER_HOSTNAME}"

    # Имя запускаемого образа:
    "$IMAGE_NAME"
    #-it $IMAGE_NAME bash
)

# Запускаем образ:
docker run "${RUNPARAMS[@]}"

# Вывод логов:
if false; then
    echo '┍━━━━━━━━━━━━━━━━━━━━━━━━┑'
    echo '│                        │'
    echo '│ Лог работы контейнера: │'
    echo '│                        │'
    echo '┕━━━━━━━━━━━━━━━━━━━━━━━━┙'
    docker logs --details -t "$DOCKER_NAME"
    echo '┍━━━━━━━━━━━━━━━━━━━━━━━━┑'
    echo '│                        │'
    echo '│       Конец лога.      │'
    echo '│                        │'
    echo '┕━━━━━━━━━━━━━━━━━━━━━━━━┙'
fi

