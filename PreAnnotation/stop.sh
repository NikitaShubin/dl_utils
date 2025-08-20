#!/usr/bin/env bash

########################
# Остановка контейнера #
########################

DOCKER_NAME='pre_annotation'

docker rm $(docker stop $(docker ps -a -q  --filter name=$DOCKER_NAME))