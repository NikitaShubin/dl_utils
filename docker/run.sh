#!/usr/bin/env bash

# Определяем положение текущего скрипта:
DOCKERFILE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Запускаемый докерфайл должен распологаться в той же дирректории.

# Задаём имя контейнера или берём его из входных параметров:
if [ $# -eq 0 ]; then
    DOCKER_NAME='dl_cv'
else
    DOCKER_NAME="$1"
    shift
fi

# Определяем имя образа:
IMAGE_NAME=kitbrain/$DOCKER_NAME

# Определяем имя хоста внутри докера:
DOCKER_HOSTNAME="${DOCKER_NAME}_`hostname`"

# Пытаемся собирать образ каждый раз заново:
#if ! docker build --progress=plain -t $IMAGE_NAME "${DOCKERFILE_DIR}"; then
if ! docker build -t $IMAGE_NAME "${DOCKERFILE_DIR}"; then
    # Если образ собрать не удалось:
    RED='\033[0;31m'
    NC='\033[0m'  # No Color (https://stackoverflow.com/a/5947802)
    printf "\n${RED}Образ не собран!\n${NC}"

    # Берём готовый с DockerHub или выводим ошибку:
    if true; then
        printf "${RED}Берётся версия из DockerHub!${NC}\n\n"
        docker pull $IMAGE_NAME
    else
        print "\n\n"
    	exit 1
    fi
else
    # Если образ успешно собран:

    # Отправляем образ на hub.docker.com:
    docker push $IMAGE_NAME
fi

#docker pull $IMAGE_NAME
# Включаем доступ к Nvidia, если установлен nvidia-smi:
nvidia-smi && nvidia_args='--runtime=nvidia --gpus all' || nvidia_args=''

# Путь до домашней папки:
home=`realpath ~`

# Параметры запуска контейнера:
RUNPARAMS=(
    # Фоновый режим:
    -d

    # Права внешнего пользователя
    # -u $(id -u):$(id -g)

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
    $nvidia_args

    # Монтируем папку проекта в докер:
    -v "${DOCKERFILE_DIR}/../../../":/workspace

    # Путь до датасета подменяем локальной его копией:
    # -v "$home/projects/AP2.0/local_files/ds/":"/workspace/data/processed"
    -v "$home/projects/IQF/local_files/ds/":"/workspace/data/processed"
    -v "/":"/outroot"
    #-v "/":"/outroot":ro

    # Имя контейнера:
    --name "${DOCKER_NAME}"

    # Пароль и порт для Jupyter:
    -e JUPYTER_PASS
    -e JUPYTER_PORT
    # Берётся из внешней переменной окружения.
    # Рекомендуется добавить строку
    # JUPYTER_PASS="my_password"; export JUPYTER_PASS
    # в файл ~/.profile, если используется bash,
    # или ~/.zshenv при использовании zsh.
    # https://unix.stackexchange.com/a/21600

    # Сетевое имя:
    -h "${DOCKER_HOSTNAME}"

    # Имя запускаемого образа:
    $IMAGE_NAME
    #-it $IMAGE_NAME bash
)

# Запускаем образ:
#clear && 
docker run "${RUNPARAMS[@]}"
#docker exec -u root -it $DOCKER_NAME bash

# Вывод логов:
if false; then
    echo '┍━━━━━━━━━━━━━━━━━━━━━━━━┑'
    echo '│                        │'
    echo '│ Лог работы контейнера: │'
    echo '│                        │'
    echo '┕━━━━━━━━━━━━━━━━━━━━━━━━┙'
    docker logs --details -t $DOCKER_NAME
    echo '┍━━━━━━━━━━━━━━━━━━━━━━━━┑'
    echo '│                        │'
    echo '│       Конец лога.      │'
    echo '│                        │'
    echo '┕━━━━━━━━━━━━━━━━━━━━━━━━┙'
fi
