#!/usr/bin/env bash

# Определяем положение текущего скрипта:
DOKERFILE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Запускаемый докерфайл должен распологаться в той же дирректории.

DOCKER_NAME='dl_cv'

echo "${DOKERFILE_DIR}"

# Создаём образ:
docker build                               \
	-t $DOCKER_NAME "${DOKERFILE_DIR}"
	#--no-cache                        \

# Запускаем образ:
docker run                                          \
	-d                                          \
	--ipc=host                                  \
	--rm                                        \
	--runtime=nvidia                            \
	--gpus all                                  \
	-p 6006:6006                                \
	-p 8888:8888                                \
	-v "${DOKERFILE_DIR}"/../../../:/workspace  \
	--name "${DOCKER_NAME}"                     \
	-h     "${DOCKER_NAME}:`cat /etc/hostname`" \
	$DOCKER_NAME
