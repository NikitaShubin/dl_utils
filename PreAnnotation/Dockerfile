FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG WORKSPACE='/workspace'       # Рабочая папка
ARG USERNAME='user'              # Имя пользователя
ARG HOMEDIR="/home/${USERNAME}"  # Домашняя папка пользователя
ARG SHELL_PATH='/bin/bash'       # Путь к интерпретатору оболочки
ARG BRC="${HOMEDIR}/.bashrc"     # Путь к настройкам сессии терминала


# Меняем sh на bash:
SHELL ["/bin/bash", "-l", "-c"]

# Чтобы apt-get не ждал ввода часового пояса:
#ENV DEBIAN_FRONTEND=noninteractive
# В данном случае не нужно, т.к. выбранный прообраз лишён этого недостатка.

# Устанавливаем доп. приложения:
ARG APPLICATIONS="git ffmpeg wget"

RUN echo $APPLICATIONS
RUN apt-get update                                           && \
    apt-get install -y --no-install-recommends $APPLICATIONS && \
    apt-get upgrade -y --fix-missing                         && \
    apt-get purge --autoremove                               && \
    apt-get autoclean                                        && \
    rm -rf /var/lib/apt/lists/*


# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8


# Устанавливаем необходимые python-библиотеки:
#ARG PIPPARAMS="--root-user-action=ignore --no-cache-dir"
ARG PIPPARAMS="--no-cache-dir"
ARG PIP="python -m pip"
ARG PYTHONMODULES="                                                      \
    # Фиксируем версию PyTorch:                                          \
    torch==2.5.1                                                         \
                                                                         \
    # Для работы с CVAT:                                                 \
    cvat-sdk==2.4.7                                                      \
                                                                         \
    # IDE:                                                               \
    jupyterlab voila                                                     \
    jupyterlab_nvdashboard                                               \
    # jupyter-collaboration                                              \
    ipympl mpl_point_clicker mpl_interactions[jupyter] matplotlib==3.9.4 \
                                                                         \
    # Базовые модули:                                                    \
    opencv-python scikit-learn pandas numba scikit-image                 \
                                                                         \
    #                    Фундаментальные модели:                         \
                                                                         \
    # HQSAM (используется в samal.py вместо Segment Anything Model):     \
    git+https://github.com/SysCV/sam-hq.git timm                         \
                                                                         \
    # Segment Anything Model 2.1 (для сегментации видео):                \
    git+https://github.com/facebookresearch/sam2.git decord              \
                                                                         \
    # Grounding DINO:                                                    \
    groundingdino-py                                                     \
    #git+https://github.com/IDEA-Research/GroundingDINO.git              \
"
# "jupyverse[jupyterlab,noauth]"
RUN ${PIP} install $PIPPARAMS --upgrade pip && \
    ${PIP} install $PIPPARAMS $PYTHONMODULES --upgrade -vvv


# Исправляем баг SAM2 с недоступностью конфигурационных файлов:
# RUN wget https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip && \
#     7z x ./main.zip && ls -laht && cd sam2-main && pip install -e ".[notebooks]" && cd .. && rm -rf sam2-main
RUN pypackagesdir=`${PIP} show sam-2 | grep Location | cut -d' ' -f2` && \
    cp -R "$pypackagesdir/sam2/configs/sam2.1/." "$pypackagesdir/sam2/" && \
    wget -P "$pypackagesdir/sam2/" \
    "https://github.com/SysCV/sam-hq/blob/main/sam-hq2/sam2/configs/sam2.1/sam2.1_hq_hiera_l.yaml"

# Отключаем у JupyterLab надоедливый вопрос об обновлении и
# выключаем поддержку voila, т.к. внтури JupyterLab она всё равно не работает:
RUN jupyter labextension disable '@jupyterlab/apputils-extension:announcements' && \
    jupyter server extension disable voila
#RUN jupyter labextension install @voila-dashboards/jupyterlab-preview
#RUN jupyter serverextension enable voila &&
#RUN 


# Точка монтирования внутри контейнера:
VOLUME  "${WORKSPACE}"
# Рабочая папка:
WORKDIR "${WORKSPACE}"
# Снимаем с рабочей папки защиту от записи:
RUN chmod -R 777 .


# Вносим нужные файлы в рабочую папку образа:
ADD --chmod=777 https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt ./models/
ADD --chmod=777 https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth ./models/
ADD --chmod=777 https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinB_cfg.py ./models/
# Самые мощные модели SAM2 и DINO (вместе с конфигом).
COPY --chown="${USERNAME}:${USERNAME}" --chmod=777 *.py ./src/
# Модули.
COPY --chown="${USERNAME}:${USERNAME}" --chmod=777 PreAnnotation/*.ipynb ./
# Интерфейсные юпитер-ноутбуки.


# Создаём пользователя, даём ему право использовать sudo без пароля и устанавливаем его по умолчанию:
RUN useradd -ms "${SHELL_PATH}" "${USERNAME}"         && \
    adduser "${USERNAME}" sudo                        && \
    echo "${USERNAME}" ALL = NOPASSWD: ALL >> /etc/sudoers
USER "${USERNAME}"

# Установка тёмной темы в JupyterLab:
ARG THEMEDIR="${HOMEDIR}"/.jupyter/lab/user-settings/\@jupyterlab/apputils-extension/
RUN mkdir -p "${THEMEDIR}" && \
    echo '{ "theme":"JupyterLab Dark" }' > "${THEMEDIR}"themes.jupyterlab-settings

# Доустановка мелочей для GroundingDINO:
RUN python -c "import sys; sys.path.append('./src'); from dinoal import Dino; dino = Dino('./models/groundingdino_swinb_cogcoor.pth')"


# Настраиваем соединение JupyterLab по-умолчанию:
ARG JPCONF="${HOMEDIR}/.jupyter/jupyter_lab_config.py"
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.allow_root = True"          >> "${JPCONF}" && \
    echo "c.ServerApp.allow_remote_access = True" >> "${JPCONF}" && \
    echo "c.ServerApp.ip = '*'"                   >> "${JPCONF}" && \
    echo "c.ServerApp.token = ''"                 >> "${JPCONF}" && \
    echo "c.ServerApp.port = 8888"                >> "${JPCONF}" && \
    echo "c.NotebookApp.open_browser = False"     >> "${JPCONF}"
# По-умолчанию пароли и токены не используются.
# Однако далее пароль будет взят из переменной окружения.


# Запуск Юпитер-сервера:
CMD JP="jupyter-lab"; \
    [[ -z "$JUPYTER_PORT" ]] || JP="$JP --port=$JUPYTER_PORT"; \
    [[ -z "$JUPYTER_PASS" ]] || JP="$JP --ServerApp.password="'"'"`echo ""$JUPYTER_PASS"" | python -c 'from jupyter_server.auth import passwd;print(passwd(input()))'`"'"'; \
    ${JP}
# Если в переменных окружения заданы пароль и порт, то они используются как параметры запуска.


# Запуск JupyterLab с указанным в переменных окружения паролем и портом:
#CMD jupyter-lab --notebook-dir="${WORKSPACE}" --port="$JUPYTER_PORT" --collaborative \
#    --NotebookApp.password="$(echo $JUPYTER_PASSWORD | python -c 'from jupyter_server.auth import passwd;print(passwd(input()))')" "${JPPARAMS}"
