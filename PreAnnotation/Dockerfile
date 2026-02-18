# hadolint global ignore = DL3008  # Не фиксируем версии, т.к. в этом образе нужны свежие пакеты
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

ARG WORKSPACE='/workspace'       # Рабочая папка
ARG USERNAME='user'              # Имя пользователя
ARG HOMEDIR="/home/${USERNAME}"  # Домашняя папка пользователя
ARG SHELL_PATH='/bin/bash'       # Путь к интерпретатору оболочки
ARG BRC="${HOMEDIR}/.bashrc"     # Путь к настройкам сессии терминала


# Меняем sh на bash:
SHELL ["/bin/bash", "-l", "-o", "pipefail", "-c"]

# Чтобы apt-get не ждал ввода часового пояса:
#ENV DEBIAN_FRONTEND=noninteractive
# В данном случае не нужно, т.к. выбранный прообраз лишён этого недостатка.

# Устанавливаем доп. приложения:
ARG APPLICATIONS="git ffmpeg wget gosu"

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
ARG PIPPARAMS="--no-cache-dir"  # Отключаем кеш
ARG PIP="python -m pip"
ARG PYTHONMODULES="                                                      \
    # Фиксируем версии PyTorch и совместимых пакетов:                    \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1                   \
                                                                         \
    # Для работы с CVAT:                                                 \
    cvat-sdk                                                             \
                                                                         \
    # IDE:                                                               \
    jupyterlab voila                                                     \
    jupyterlab_nvdashboard                                               \
    # jupyter-collaboration                                              \
    ipympl mpl_point_clicker mpl_interactions[jupyter] matplotlib        \
    Jupyter-Beeper                                                       \
                                                                         \
    # Базовые модули:                                                    \
    opencv-python scikit-learn pandas numba scikit-image                 \
                                                                         \
    # Для конвертации меток:                                             \
    treelib openpyxl                                                     \
                                                                         \
    # Коллекция YOLO-моделей:                                            \
    ultralytics lapx git+https://github.com/ultralytics/CLIP.git         \
    # CLIP нужен для работы SAM3 в ultralytics                           \
                                                                         \
    # Трекеры:                                                           \
    boxmot                                                               \
                                                                         \
    #                    Фундаментальные модели:                         \
                                                                         \
    # HQSAM (используется в samal.py вместо Segment Anything Model):     \
    # git+https://github.com/SysCV/sam-hq.git timm                       \
    segment-anything-hq                                                  \
                                                                         \
    # Segment Anything Model 2.1 (для сегментации видео):                \
    git+https://github.com/facebookresearch/sam2.git decord              \
                                                                         \
    # Segment Anything Model 3:                                          \
    # https://github.com/facebookresearch/sam3.git                       \
    sam3                                                                 \
                                                                         \
    # Grounding DINO:                                                    \
    groundingdino-py                                                     \
                                                                         \
    # Китайский аналог huggingface без заморочек с правами:              \
    modelscope                                                           \
"

# Сначала обновляем сам pip.
# Затем устанавливаем пакеты БЕЗ зависимостей, на случай проблем.
# Потом устанавливаем все зависимости для ранее установленных пакетов
# Наконец, очищаем кэш pip для уменьшения размера образа
RUN ${PIP} install $PIPPARAMS --upgrade pip && \
    eval "${PIP} install $PIPPARAMS $PYTHONMODULES --upgrade" && \
    eval "${PIP} cache purge"

# Отключаем у JupyterLab надоедливый вопрос об обновлении и
# выключаем поддержку voila, т.к. внтури JupyterLab она всё равно не работает:
RUN jupyter labextension disable '@jupyterlab/apputils-extension:announcements' && \
    jupyter server extension disable voila

# Точка монтирования внутри контейнера:
VOLUME  "${WORKSPACE}"
# Рабочая папка:
WORKDIR "${WORKSPACE}"
# Снимаем с рабочей папки защиту от записи:
RUN chmod -R 777 .

COPY --chown="${USERNAME}:${USERNAME}" --chmod=777 *.py ./src/
# Модули.
COPY --chown="${USERNAME}:${USERNAME}" --chmod=777 PreAnnotation/*.ipynb PreAnnotation/*.xlsx ./
# Интерфейсные юпитер-ноутбуки.

# Создаём пользователя и даём ему право использовать sudo без пароля:
RUN useradd -ms "${SHELL_PATH}" "${USERNAME}"         && \
    adduser "${USERNAME}" sudo                        && \
    echo "${USERNAME}" ALL = NOPASSWD: ALL >> /etc/sudoers

# Скачиваем модели SAM2, DINO, YOLO и ReID в ~/models/:
RUN python ./src/sam2al.py                          && \
    mv /root/models "${HOMEDIR}"                    && \
    # Модель для SAM2 приходится качать под root из-за необходимости править сам модуль sam2.
    chown -R "${USERNAME}:${USERNAME}" "${HOMEDIR}" && \
    # Все файлы в домашней папке пользователя должны снова пренадлежать пользователю.
    gosu "${USERNAME}" python ./src/gdinoal.py      && \
    gosu "${USERNAME}" python ./src/ul_utils.py     && \
    gosu "${USERNAME}" python ./src/boxmot_utils.py
    # Остальные модели можно качать под обычным пользователем.

# Устанавливаем пользователя по умолчанию:
USER "${USERNAME}"

# Установка тёмной темы в JupyterLab:
ARG THEMEDIR="${HOMEDIR}"/.jupyter/lab/user-settings/\@jupyterlab/apputils-extension/
RUN mkdir -p "${THEMEDIR}" && \
    echo '{ "theme":"JupyterLab Dark" }' > "${THEMEDIR}"themes.jupyterlab-settings


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
CMD ["/bin/bash", "-c", " \
    JP=\"jupyter-lab\"; \
    [[ -z \"$JUPYTER_PORT\" ]] || JP=\"$JP --port=$JUPYTER_PORT\"; \
    [[ -z \"$JUPYTER_PASS\" ]] || JP=\"$JP --ServerApp.password=$(echo \"$JUPYTER_PASS\" | python -c 'from jupyter_server.auth import passwd;print(passwd(input()))')\"; \
    exec $JP \
"]
# Если в переменных окружения заданы пароль и порт, то они используются как параметры запуска.
