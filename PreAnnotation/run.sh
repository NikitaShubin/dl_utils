#!/usr/bin/env bash

#########################
# Перезапуск контейнера #
#########################

# Определяем положение текущего скрипта:
DOCKERFILE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Запускаемый докерфайл должен распологаться в той же дирректории.

# Задаём имя контейнера или берём его из входных параметров:
if [ $# -eq 0 ]; then
    DOCKER_NAME='pre_annotation'
else
    DOCKER_NAME="$1"
    shift
fi

# Определяем имя образа:
IMAGE_NAME=kitbrain/$DOCKER_NAME

# Определяем имя хоста внутри докера:
DOCKER_HOSTNAME="${DOCKER_NAME}_`hostname`"

# Определяем теги образа:
GITTAG=GIT-$(git rev-parse --short HEAD)
DATETAG=$(date +"%Y.%m.%d")

# Строка образа с тегами:
IMAGE_NAME_AND_TAGS=(
    -t "$IMAGE_NAME:$GITTAG"
    -t "$IMAGE_NAME:$DATETAG"
    -t "$IMAGE_NAME:latest"
)

# Перемещаемся на уровень выше от докер-файла:
cur_dir=`pwd`
echo $cur_dir
cd "${DOCKERFILE_DIR}"/..


# Пытаемся собирать образ каждый раз заново:
# if ! docker build --progress=plain "${IMAGE_NAME_AND_TAGS[@]}" -f "$DOCKERFILE_DIR/Dockerfile" .;
if ! docker build "${IMAGE_NAME_AND_TAGS[@]}" -f "$DOCKERFILE_DIR/Dockerfile" .;
then
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
    # Если образ успешно собран, отправляем ВСЕ теги в одном фоновом процессе:
    nohup bash -c "
        echo 'Начало отправки образов'
        docker push '$IMAGE_NAME:$GITTAG'
        docker push '$IMAGE_NAME:$DATETAG'
        docker push '$IMAGE_NAME:latest'
        echo 'Отправка завершена'
    " > /dev/null 2>&1 &
    echo "Запущена отправка всех тегов в фоне (PID: $!)"
fi

# Возвращаем исходное значение текущей папки:
cd "${cur_dir}"

#docker pull $IMAGE_NAME
# Включаем доступ к Nvidia, если установлен nvidia-smi:
nvidia-smi && nvidia_args='--runtime=nvidia --gpus all' || nvidia_args=''

# Параметры запуска контейнера:
JUPYTER_PORT=573
RUNPARAMS=(
    # Фоновый режим:
    -d

    # Удаляем контейнер после завершения:
    --rm
    # Не совместим с --restart unless-stopped.

    # Автоматический перезапуск вместе с демоном, если
    # к моменту перезапуска контейнер не был остановлен вручную:
    # --restart unless-stopped
    # Не совместим с --rm.

    # Имя контейнера:
    --name "${DOCKER_NAME}"

    # Сетевое имя:
    -h "${DOCKER_HOSTNAME}"

    -e JUPYTER_PASS=PreAnnotator
    -e JUPYTER_PORT=$JUPYTER_PORT

    # Пробрасываем порт без изменений:
    -p $JUPYTER_PORT:$JUPYTER_PORT

    # Монтируем папку проекта в докер:
    -v "${DOCKERFILE_DIR}/project/":/workspace/project
    -v "/":"/outroot":ro
    # Внешнюю файловую систему монтируем только для чтения в одну из корневых папок.

    # Включаем доступ к GPU, если возможно:
    $nvidia_args

    # Имя запускаемого образа:
    $IMAGE_NAME
    #-it $IMAGE_NAME bash
)

# Запускаем образ:
docker run "${RUNPARAMS[@]}"