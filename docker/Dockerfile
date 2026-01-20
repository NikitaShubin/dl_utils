# hadolint global ignore = DL3008, DL3016  # Не фиксируем версии, т.к. в этом образе нужны свежие пакеты
ARG TFVER='2.19.0'
FROM tensorflow/tensorflow:${TFVER}-gpu
#FROM tensorflow/tensorflow:latest-gpu
#FROM tensorflow/tensorflow:nightly-gpu


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
ARG APPLICATIONS="                                    \
    # Утилиты для работы с пакетами:                  \
    apt-utils                                         \
                                                      \
    # Линтеры:                                        \
    shellcheck                                        \
                                                      \
    # Для работы в Python:                            \
    graphviz pandoc                                   \
                                                      \
    # Утилиты для мониторинга ресурсов:               \
    htop iotop nvtop nethogs                          \
                                                      \
    # Утилиты для работы с мультимедиа:               \
    ffmpeg yt-dlp imagemagick                         \
                                                      \
    # Базовые пользовательские приложения в консоли:  \
    sudo gosu nano mc git ssh p7zip-full zip unzip    \
                                                      \
    # Консольные утилиты для загрузки файлов из сети: \
    curl wget wget2                                   \
                                                      \
    # Пакеты для работы manim:                        \
    python3-dev libcairo2-dev libpango1.0-dev         \
                                                      \
    # LaTeX-компилятор:                               \
    texlive-full                                      \
                                                      \
    # Дополнительные консольные утилиты:              \
    bsdmainutils pv net-tools psmisc iputils-ping     \
    tree file coreutils                               \
                                                      \
    # Для PyCUDA:                                     \
    nvidia-cuda-toolkit                               \
"
ARG HADOLINTURL="https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64"
# graphviz     - для работы keras.utils.plot_model
# pandoc       - для работы экспорта ipynb в HTML
# nethogs      - мониторинг загрузки сети
# iotop        - мониторинг использования накопителей
# nvtop        - мониторинг использования видюх от Nvidia
# yt-dlp       - для загрузки видео/медиа-файлов с Youtube и не только
# psmisc       - для работы команды killall
# net-tools    - для работы команды ifconfig
# bsdmainutils - для работы команды col
# iputils-ping - для работы команды ping
# coreutils    - для работы команды wc
RUN apt-get update                                            && \
    # Удаление python3-blinker необходимо для нормальной установки open3d:
    apt-get purge -y python3-blinker                          && \
    apt-get install -y --no-install-recommends $APPLICATIONS  && \
    apt-get upgrade -y --fix-missing                          && \
    apt-get purge -y --autoremove                             && \
    apt-get autoclean                                         && \
    rm -rf /var/lib/apt/lists/*                               && \
    # Установка Hadolint и markdownlint-cli:
    wget --progress=dot:giga -O /bin/hadolint $HADOLINTURL    && \
    chmod +x /bin/hadolint                                    && \
    # Установка Node.js из NodeSource (вместо старых версий из репозитория):
    wget -qO- https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y --no-install-recommends nodejs         && \
    npm install -g markdownlint-cli dclint

# См. http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Устанавливаем необходимые python-библиотеки:
#ARG PIPPARAMS="--root-user-action=ignore --no-cache-dir"
ARG PIPPARAMS="--root-user-action=ignore"
ARG PIP="python -m pip"
ARG PYTHONMODULES_BASE="                                                 \
    # Keras и прочее:                                                    \
    keras keras-tuner pydot tf_keras keras_hub                           \
                                                                         \
    # Катбуст:                                                           \
    catboost                                                             \
                                                                         \
    # Мелкие полезные утилиты:                                           \
    treelib openpyxl easydict                                            \
                                                                         \
    # Aугментация и прочая обработка изображений + доп. утилиты:         \
    albumentations scikit-learn scikit-image                             \
                                                                         \
    # Работа с облаком точек и картой глубины:                           \
    open3d                                                               \
                                                                         \
    # YOLO и трекинг в ней + Roboflow:                                   \
    ultralytics lapx supervision                                         \
                                                                         \
    # Трекеры:                                                           \
    stonesoup                                                            \
                                                                         \
    # Большой банк предобученных CV-моделей                              \
    git+https://github.com/leondgarse/keras_cv_attention_models          \
                                                                         \
    # Torch:                                                             \
    torchsummary torchmetrics                                            \
                                                                         \
    # Дополнительно для CV:                                              \
    kornia                                                               \
                                                                         \
    # Математическая анимация:                                           \
    manim                                                                \
                                                                         \
                                                                         \
    # Для работы с CVAT:                                                 \
    cvat_sdk==2.45                                                       \
    #cvat-sdk[masks,pytorch]                                             \
                                                                         \
    # IDE (JupyterLab\Jupyverse и PyCharm):                              \
    jupyterlab voila xeus-python                                         \
    jupyterlab_nvdashboard jupyter-resource-usage                        \
    jupyterlab-latex jupyterlab-git                                      \
    ipympl mpl_point_clicker mpl_interactions[jupyter]                   \
    Jupyter-Beeper                                                       \
    # jupyter-collaboration jupyterlab_autoscrollcelloutput              \
    # pydevd-pycharm jupyverse[jupyterlab,auth]                          \
    # ipympl, mpl_point_clicker и mpl_interactions для интерактивного    \
    # matplotlib в ноутбуках. С их помощью можно, например, выполнять    \
    # интерактивную сегментацию с SAM через ноутбук.                     \
    # xeus-python для отладки.                                           \
                                                                         \
    # Отключает бОльшую часть мусора в stdout, генерируемого TensorFlow: \
    silence-tensorflow tensorflow-model-optimization                     \
    # tensorflow-tensorboard==1.5.1 tensorboard==1.15.0                  \
    # tensorflow_hub tensorflow_datasets jupyter-tensorboard             \
                                                                         \
    # Всё для ONNX (используется в onnx_utils.py):                       \
    onnx onnxmltools onnxconverter-common                                \
    git+https://github.com/onnx/tensorflow-onnx                          \
                                                                         \
    onnxruntime-gpu --extra-index-url  https://aiinfra.pkgs.visualstudio.\
com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/           \
    # Доп. параметр необходим для работы onnxruntime на gpu при          \
    # использовании CUDA 12 (см. https://onnxruntime.ai/docs/install/).  \
                                                                         \
    # Линтеры, стайлчекеры, тестеры и прочие утилиты работы с кодом:     \
    cookiecutter-data-science pytest-cov pycodestyle pycodestyle_magic   \
    python-lsp-server[all] jupyterlab-lsp black[jupyter] ruff flake8     \
    isort[requirements_deprecated_finder,pipfile_deprecated_finder]      \
    nbqa pylint wemake-python-styleguide ondivi mypy pandas-stubs        \
    # pandas-stubs нужен для работы с pandas в mypy                      \
                                                                         \
                                                                         \
    # Python-компилятор                                                  \
    pyinstaller                                                          \
                                                                         \
    # Ускорение Python-кода:                                             \
    numba                                                                \
                                                                         \
    #                    Фундаментальные модели:                         \
                                                                         \
    # DEVA (Tracking Anything with Decoupled Video Segmentation):        \
    # git+https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git  \
                                                                         \
    # HQSAM (используется в samal.py вместо Segment Anything Model):     \
    git+https://github.com/SysCV/sam-hq.git timm                         \
                                                                         \
    # Segment Anything Model 2.1 (для сегментации видео):                \
    # git+https://github.com/facebookresearch/sam2.git decord            \
    sam2 decord                                                          \
                                                                         \
    # Segment Anything Model 3:                                          \
    # https://github.com/facebookresearch/sam3.git                       \
    sam3                                                                 \
                                                                         \
    # Language Segment-Anything:                                         \
    # git+https://github.com/luca-medeiros/lang-segment-anything.git     \
                                                                         \
    # Grounding DINO:                                                    \
    groundingdino-py                                                     \
                                                                         \
    # Гигантская библиотека предобученных моделей:                       \
    # git+https://github.com/huggingface/transformers.git                \
    # Нужна более старая версия для работы Grounding DINO                \
                                                                         \
    # Вейвлеты:                                                          \
    PyWavelets                                                           \
                                                                         \
    #                            Прочее:                                 \
                                                                         \
    pycuda                                                               \
"
ARG PYTHONMODULES_EXTRA="                                                \
    # Трекеры:                                                           \
    boxmot regex>=2025.11.3                                              \

    # Навороты в Jupyter:                                                \
    jupyter-ai[all]                                                      \
    # jupyter-ai для использования API всяких LLM-ок.                    \
"
    # Сначала обновляем сам pip:
RUN ${PIP} install $PIPPARAMS --upgrade pip && \
    # Затем фиксируем текущую версию уже установленного Tensorflow:
    TFVER=$(${PIP} show tensorflow | grep Vers | awk '{print $2}') && \
    # И устанавливаем основные пакеты:
    eval "${PIP} install $PIPPARAMS tensorflow==${TFVER} $PYTHONMODULES_BASE --upgrade" && \
    # Потом устанавливаем все зависимости для ранее установленных пакетов:
    eval "${PIP} install $PIPPARAMS $PYTHONMODULES_EXTRA" && \
    # Наконец, очищаем кэш pip для уменьшения размера образа:
    eval "${PIP} cache purge" && \
    # Приходится пересобирать Jupyter Lab, т.к. установка Node.js из NodeSource приводит к поломкам зависимостей:
    jupyter lab build --minimize=False

# Установка всех необходимых пакетов при фиксации версии уже установленного Tensorflow.
#    ${PIP} install $PIPPARAMS tensorflow==${TFVER} $PYTHONMODULES --upgrade --no-deps && \
#    ${PIP} install $PIPPARAMS $PYTHONMODULES && \

# Отключаем у JupyterLab надоедливый вопрос об обновлении
RUN jupyter labextension disable '@jupyterlab/apputils-extension:announcements'

# Создаём пользователя и даём ему право использовать sudo без пароля:
RUN useradd -ms "${SHELL_PATH}" "${USERNAME}"              && \
    adduser "${USERNAME}" sudo                             && \
    echo "${USERNAME}" ALL = NOPASSWD: ALL >> /etc/sudoers && \
    chmod 1777 /etc && chmod 1777 "${HOMEDIR}"

# Скачиваем модели SAM2 и DINO в ~/models/:
COPY *.py /tmp/py/
RUN python ./tmp/py/sam2al.py                       && \
    mv /root/models "${HOMEDIR}"                    && \
    # Модель для SAM2 приходится качать под root из-за необходимости править сам модуль sam2.
    chown -R "${USERNAME}:${USERNAME}" "${HOMEDIR}" && \
    # Все файлы в домашней папке пользователя должны снова пренадлежать пользователю.
    gosu "${USERNAME}" python ./tmp/py/gdinoal.py
    # Остальные модели можно качать под обычным пользователем.

# Переключаемся на пользователя:
USER "${USERNAME}"

# Выключаем поддержку voila, т.к. внтури JupyterLab она всё равно не работает:
#RUN jupyter server extension disable voila
#RUN jupyter labextension install @voila-dashboards/jupyterlab-preview
#RUN jupyter serverextension enable voila &&

# Грязно отключаем support_message, используемый в silence-tensorflow, т.к. он работает некорректно без root-a:
#RUN echo 'def support_message(*args, **kwargs): pass' > `python -c \
#    'import inspect, support_developer; print(inspect.getfile(support_developer.support_message))'`

# Установка тёмной темы в JupyterLab:
ARG THEMEDIR="${HOMEDIR}"/.jupyter/lab/user-settings/\@jupyterlab/apputils-extension/
RUN mkdir -p "${THEMEDIR}" && \
    echo '{ "theme":"JupyterLab Dark" }' > "${THEMEDIR}"themes.jupyterlab-settings

# Загрузка весов свёрточной части предобученной модели EfficientNetV2M:
#ARG KERAS_MODELS_DIR="${HOMEDIR}/.keras/models/"
#ARG KERAS_APP_URL='https://storage.googleapis.com/tensorflow/keras-applications/'
#RUN wget --progress=dot:giga -P "${KERAS_MODELS_DIR}" "${KERAS_APP_URL}efficientnet_v2/efficientnetv2-m_notop.h5"

# Убираем надоедливое 
# "/sbin/ldconfig.real: Can't create temporary cache file /etc/ld.so.cache~: Permission denied"
# при запуске:
#RUN sudo chmod u+s /sbin/ldconfig.real

# Задаём текстовый редактор по умолчанию для mc:
RUN echo SELECTED_EDITOR=\"/bin/nano\"> "${HOMEDIR}"/.selected_editor

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

# Другине настройки Jupyter-сервера:
RUN echo "c.ContentsManager.allow_hidden = True"  >> "${JPCONF}" && \
    echo "c.LatexConfig.run_times = 2"            >> "${JPCONF}" && \
    echo "c.LatexConfig.manual_cmd_args = ["      >> "${JPCONF}" && \
    echo "    'pdflatex',"                        >> "${JPCONF}" && \
    echo "    '-interaction=nonstopmode',"        >> "${JPCONF}" && \
    echo "    '-halt-on-error',"                  >> "${JPCONF}" && \
    echo "    '-file-line-error'"                 >> "${JPCONF}" && \
    echo "    '-shell-escape',"                   >> "${JPCONF}" && \
    echo "    '-synctex=1',"                      >> "${JPCONF}" && \
    echo "    '{filename}.tex'"                   >> "${JPCONF}" && \
    echo "]"                                      >> "${JPCONF}"
# Отображение скрытых файлов в файловом менеджере.
# Двойной запуск компиляции LaTeX для проработки всяких ссылок.
# Прочие конфигурации LaTeX.


# Взято с https://gist.github.com/jbwhit/eecdd1cac2756df85ad165f437445b0b.

#RUN export jupass=`python -c "from jupyter_server.auth import passwd; print(passwd('test'))"` && \
#    echo "c.NotebookApp.password = '${jupass}'"     >> "${JPCONF}"


# Поддержка LaTeX в JupyterLab:
#RUN jupyter lab --generate-config && \
#    echo "c.NotebookApp.extra_static_paths = ['node_modules/mathjax/es5/tex-mml-chtml.js']" >> ~/.jupyter/jupyter_lab_config.py

# Прописываем ряд полезных алиасов в "${BRC}":
#ARG JP="${HOMEDIR}"/.local/bin/jupyter-lab
ARG JP=jupyter-lab
ARG JPPARAMS="--notebook-dir '${WORKSPACE}'"
RUN echo "alias jp=\"cd '${WORKSPACE}' && ${JP} ${JPPARAMS} > /dev/null &\""            >> "${BRC}" && \
    echo "alias jl=\"${JP} list\""                                                      >> "${BRC}" && \
    echo "alias js=\"killall jupyter-lab\""                                             >> "${BRC}" && \
    echo "alias ds=\"js\""                                                              >> "${BRC}" && \
                                                                                                       \
    echo "alias checkcrc=\"md5sum -c crc.md5\""                                         >> "${BRC}" && \
    echo "alias makecrc=\"find . -type f ! -name crc.md5 -exec md5sum {} + > crc.md5\"" >> "${BRC}" && \
                                                                                                       \
    echo "alias yt=\"yt-dlp\""                                                          >> "${BRC}" && \
    echo "alias ffmpeg=/usr/bin/ffmpeg"                                                 >> "${BRC}" && \
    echo "alias ffprobe=/usr/bin/ffprobe"                                               >> "${BRC}" && \
    echo "alias ffplay=/usr/bin/ffplay"                                                 >> "${BRC}" && \
                                                                                                       \
    echo "alias nv=\"nvidia-smi -l 1\""                                                 >> "${BRC}" && \
                                                                                                       \
    echo "alias brc=\"nano '${BRC}' && source '${BRC}'\""                               >> "${BRC}" && \
                                                                                                       \
    echo "eval \"\$(ssh-agent -s)\" > /dev/null"                                        >> "${BRC}" && \
    echo "ssh-add -q ~/.ssh/* 2> /dev/null"                                             >> "${BRC}"


# Используемые порты:
EXPOSE 8888 6006 8000 8866
# JupyterLab, TensorBoard, KerasTuner, Voila.

# Точка монтирования внутри контейнера:
VOLUME  "${WORKSPACE}"
# Рабочая папка:
WORKDIR "${WORKSPACE}"

# Вносим рабочую дирректорию в список доверенных для Git:
RUN git config --global --add safe.directory "$WORKSPACE"                     && \
    git config --global --add safe.directory "$WORKSPACE"/3rdparty/dl_utils   && \
    git config --global --add safe.directory "$WORKSPACE"/thirdparty/dl_utils && \
    git config --global core.fileMode false                                   && \
    git config --global user.email shubin.kit@ya.ru                           && \
    git config --global user.name "Никита Шубин"

# Запуск Юпитер-сервера:
CMD ["/bin/bash", "-c", " \
    JP=\"jupyter lab\"; \
    [[ -z \"$JUPYTER_PORT\" ]] || JP=\"$JP --port=$JUPYTER_PORT\"; \
    [[ -z \"$JUPYTER_PASS\" ]] || JP=\"$JP --ServerApp.password=$(echo \"$JUPYTER_PASS\" | python -c 'from jupyter_server.auth import passwd;print(passwd(input()))')\"; \
    python */dl_utils/ollm_utils.py; \
    exec $JP \
"]
# Если в переменных окружения заданы пароль и порт, то они используются как параметры запуска.
# ollm_utils автоматически конфигурирует jupyter-ai.
