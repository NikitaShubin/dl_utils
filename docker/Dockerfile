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
SHELL ["/bin/bash", "-l", "-c"]

# Чтобы apt-get не ждал ввода часового пояса:
#ENV DEBIAN_FRONTEND=noninteractive
# В данном случае не нужно, т.к. выбранный прообраз лишён этого недостатка.

# Устанавливаем доп. приложения:
ARG APPLICATIONS="                                    \
    # Утилиты для работы с пакетами:                  \
    apt-utils                                         \
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
    sudo nano mc git ssh p7zip-full zip unzip         \
                                                      \
    # Консольные утилиты для загрузки файлов из сети: \
    curl wget                                         \
                                                      \
    # Пакеты для работы manim:                        \
    python3-dev libcairo2-dev libpango1.0-dev         \
                                                      \
    # LaTeX-компилятор:                               \
    texlive-full                                      \
                                                      \
    # Дополнительные консольные утилиты:              \
    bsdmainutils pv net-tools psmisc iputils-ping     \
    tree                                              \
                                                      \
    # Для PyCUDA:                                     \
    nvidia-cuda-toolkit                               \
"
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
RUN apt-get update                                           && \
    apt-get purge -y python3-blinker                         && \
    apt-get install -y --no-install-recommends $APPLICATIONS && \
    apt-get upgrade -y --fix-missing                         && \
    apt-get purge -y --autoremove                            && \
    apt-get autoclean                                        && \
    rm -rf /var/lib/apt/lists/*
# Удаление python3-blinker необходимо для нормальной установки open3d

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Устанавливаем необходимые python-библиотеки:
#ARG PIPPARAMS="--root-user-action=ignore --no-cache-dir"
ARG PIPPARAMS=""
ARG PIP="python -m pip"
ARG PYTHONMODULES="                                                      \
    # Keras и прочее:                                                    \
    keras keras-tuner pydot tf_keras keras_hub                           \
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
    ultralytics==8.3.228 lapx supervision                                \
                                                                         \
    # Большой банк предобученных CV-моделей                              \
    git+https://github.com/leondgarse/keras_cv_attention_models          \
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
    ipympl mpl_point_clicker mpl_interactions[jupyter] matplotlib        \
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
    cookiecutter-data-science pytest pycodestyle pycodestyle_magic       \
    python-lsp-server[all] jupyterlab-lsp black[jupyter] ruff            \
    isort[requirements_deprecated_finder,pipfile_deprecated_finder]      \
    flake8 nbqa pylint wemake-python-styleguide ondivi                   \
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
    #                            Прочее:                                 \
                                                                         \
    pycuda                                                               \
"

# Сначала обновляем сам pip.
# Затем фиксируем текущую версию уже установленного Tensorflow и устанавливаем остальные пакеты БЕЗ зависимостей, на случай проблем.
# Потом устанавливаем все зависимости для ранее установленных пакетов.
# Наконец, очищаем кэш pip для уменьшения размера образа.
RUN ${PIP} install $PIPPARAMS --upgrade pip && \
    export TFVER=`${PIP} show tensorflow | grep Vers | awk '{print $2}'` && \
    ${PIP} install $PIPPARAMS tensorflow==${TFVER} $PYTHONMODULES --upgrade && \
    ${PIP} cache purge
# Установка всех необходимых пакетов при фиксации версии уже установленного Tensorflow.
#    ${PIP} install $PIPPARAMS tensorflow==${TFVER} $PYTHONMODULES --upgrade --no-deps && \
#    ${PIP} install $PIPPARAMS $PYTHONMODULES && \


# Исправляем баг SAM2 с недоступностью конфигурационных файлов:
# RUN wget https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip && \
#     7z x ./main.zip && ls -laht && cd sam2-main && pip install -e ".[notebooks]" && cd .. && rm -rf sam2-main
#RUN pypackagesdir=`${PIP} show sam-2 | grep Location | cut -d' ' -f2` && \
RUN pypackagesdir=`${PIP} show sam2 | grep Location | cut -d' ' -f2` && \
    cp -R "$pypackagesdir/sam2/configs/sam2.1/." "$pypackagesdir/sam2/" && \
    wget -P "$pypackagesdir/sam2/" \
    "https://github.com/SysCV/sam-hq/blob/main/sam-hq2/sam2/configs/sam2.1/sam2.1_hq_hiera_l.yaml"

# Отключаем у JupyterLab надоедливый вопрос об обновлении
RUN jupyter labextension disable '@jupyterlab/apputils-extension:announcements'

# Создаём пользователя, даём ему право использовать sudo без пароля и устанавливаем его по умолчанию:
RUN useradd -ms "${SHELL_PATH}" "${USERNAME}"         && \
    adduser "${USERNAME}" sudo                        && \
    echo "${USERNAME}" ALL = NOPASSWD: ALL >> /etc/sudoers
USER "${USERNAME}"


# Выключаем поддержку voila, т.к. внтури JupyterLab она всё равно не работает:
#RUN jupyter server extension disable voila
#RUN jupyter labextension install @voila-dashboards/jupyterlab-preview
#RUN jupyter serverextension enable voila &&

# Грязно отключаем support_message, используемый в silence-tensorflow, т.к. он работает некорректно без root-a:
#RUN echo 'def support_message(*args, **kwargs): pass' > `python -c \
#    'import inspect, support_developer; print(inspect.getfile(support_developer.support_message))'`

# Скачиваем самые мощные модели SAM2 и DINO:
#RUN wget -c --retry-connrefused -t 10 -T 900 -P "${HOMEDIR}"/models/ \
#    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
#    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt "${HOMEDIR}"/models/
ADD https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth "${HOMEDIR}"/models/
RUN sudo chmod -R 777 "${HOMEDIR}"/models/

# Доустановка мелочей для GroundingDINO:
RUN python -c "import os; import groundingdino; from groundingdino.util.inference import load_model; load_model(os.path.join(os.path.dirname(os.path.abspath(groundingdino.__file__)), 'config', 'GroundingDINO_SwinB_cfg.py'), '${HOMEDIR}/models/groundingdino_swinb_cogcoor.pth')"


# Установка тёмной темы в JupyterLab:
ARG THEMEDIR="${HOMEDIR}"/.jupyter/lab/user-settings/\@jupyterlab/apputils-extension/
RUN mkdir -p "${THEMEDIR}" && \
    echo '{ "theme":"JupyterLab Dark" }' > "${THEMEDIR}"themes.jupyterlab-settings

# Загрузка весов свёрточной части предобученной модели EfficientNetV2M:
#ARG KERAS_MODELS_DIR="${HOMEDIR}/.keras/models/"
#ARG KERAS_APP_URL='https://storage.googleapis.com/tensorflow/keras-applications/'
#RUN wget -P "${KERAS_MODELS_DIR}" "${KERAS_APP_URL}efficientnet_v2/efficientnetv2-m_notop.h5"

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
RUN echo "c.LatexConfig.run_times = 2"           >> "${JPCONF}" && \
    echo "c.ContentsManager.allow_hidden = True" >> "${JPCONF}"
# Двойной запуск компиляции LaTeX для проработки всяких ссылок.
# Отображение скрытых файлов в файловом менеджере.


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
RUN echo 'alias jp="'"cd '${WORKSPACE}' && '${JP}' ${JPPARAMS} > /dev/null &"'"'      >> "${BRC}" && \
    echo 'alias jl="'${JP}' list"'                                                    >> "${BRC}" && \
    echo 'alias js="killall jupyter-lab"'                                             >> "${BRC}" && \
    echo 'alias ds="js"'                                                              >> "${BRC}" && \
                                                                                                     \
    echo 'alias checkcrc="md5sum -c crc.md5'                                          >> "${BRC}" && \
    echo 'alias makecrc="find . -type f ! -name crc.md5 -exec md5sum {} + > crc.md5"' >> "${BRC}" && \
                                                                                                     \
    echo 'alias yt="yt-dlp"'                                                          >> "${BRC}" && \
    echo 'alias ffmpeg=/usr/bin/ffmpeg'                                               >> "${BRC}" && \
    echo 'alias ffprobe=/usr/bin/ffprobe'                                             >> "${BRC}" && \
    echo 'alias ffplay=/usr/bin/ffplay'                                               >> "${BRC}" && \
                                                                                                     \
    echo 'alias nv="nvidia-smi -l 1"'                                                 >> "${BRC}" && \
                                                                                                     \
    echo 'alias brc="nano '"${BRC}"' && source '"${BRC}"'"'                           >> "${BRC}"


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
CMD JP="jupyter lab"; \
    [[ -z "$JUPYTER_PORT" ]] || JP="$JP --port=$JUPYTER_PORT"; \
    [[ -z "$JUPYTER_PASS" ]] || JP="$JP --ServerApp.password="'"'"`echo ""$JUPYTER_PASS"" | python -c 'from jupyter_server.auth import passwd;print(passwd(input()))'`"'"'; \
    ${JP}
# Если в переменных окружения заданы пароль и порт, то они используются как параметры запуска.