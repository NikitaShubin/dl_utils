#!/usr/bin/env bash

DOCKER_NAME='video_pre_annotation'

docker rm $(docker stop $(docker ps -a -q  --filter name=$DOCKER_NAME))