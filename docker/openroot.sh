#!/usr/bin/env bash

#######################################
# Вход в консоль контейнера под рутом #
#######################################

# Задаём имя контейнера или берём его из входных параметров:
if [ $# -eq 0 ]; then
    DOCKER_NAME="dl_cv_$USER"
else
    DOCKER_NAME="$1"
    shift
fi

docker exec -u root -it $DOCKER_NAME bash