ARG TFVER='2.16.1'
FROM tensorflow/tensorflow:${TFVER}-gpu
#FROM tensorflow/tensorflow:2.14.0-gpu
#FROM tensorflow/tensorflow:2.15.0-gpu
#FROM tensorflow/tensorflow:2.16.1-gpu
#FROM tensorflow/tensorflow:2.17.0-gpu
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
    sudo nano mc git ssh p7zip-full                   \
                                                      \
    # Консольные утилиты для загрузки файлов из сети: \
    curl wget                                         \
                                                      \
    # Пакеты для работы manim:                        \
    python3-dev libcairo2-dev libpango1.0-dev         \
                                                      \
    # LaTeX-компилятор:                               \
    texlive-xetex                                     \
                                                      \
    # Дополнительные консольные утилиты:              \
    bsdmainutils pv net-tools psmisc iputils-ping     \
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
    # Keras и прочее:                                                    \
    keras keras-tuner pydot tf_keras                                     \
                                                                         \
    # Мелкие полезные утилиты:                                           \
    treelib openpyxl easydict                                            \
                                                                         \
    # Aугментация и прочая обработка изображений + доп. утилиты:         \
    albumentations scikit-learn scikit-image                             \
                                                                         \
    # YOLO и трекинг в ней + Roboflow:                                   \
    ultralytics lapx supervision                                         \
                                                                         \
    # Большой банк предобученных CV-моделей                              \
    git+https://github.com/leondgarse/keras_cv_attention_models          \
                                                                         \
    # Математическая анимация:                                           \
    manim                                                                \
                                                                         \
                                                                         \
    # Для работы с CVAT:                                                 \
    cvat-sdk[masks,pytorch]==2.4.7                                       \
                                                                         \
    # IDE (JupyterLab\Jupyverse и PyCharm):                              \
    jupyterlab voila xeus-python                                         \
    jupyterlab_nvdashboard jupyter-resource-usage                        \
    jupyterlab-latex jupyterlab-git                                      \
    ipympl mpl_point_clicker mpl_interactions[jupyter] matplotlib        \
    jupyter-collaboration jupyterlab_autoscrollcelloutput                \
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
    onnx tf2onnx onnxmltools onnxconverter-common                        \
                                                                         \
    onnxruntime-gpu --extra-index-url  https://aiinfra.pkgs.visualstudio.\
com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/           \
    # Доп. параметр необходим для работы onnxruntime на gpu при          \
    # использовании CUDA 12 (https://onnxruntime.ai/docs/install/).      \
                                                                         \
    # Линтеры, стайлчекеры, тестеры и прочие утилиты работы с кодом:     \
    cookiecutter-data-science pytest pycodestyle pycodestyle_magic       \
    python-lsp-server[all] jupyterlab-lsp black[jupyter]                 \
    isort[requirements_deprecated_finder,pipfile_deprecated_finder]      \
    flake8 nbqa pylint wemake-python-styleguide ondivi                   \
                                                                         \
                                                                         \
    # Python-компилятор                                                  \
    pyinstaller                                                          \
                                                                         \
    # Для мониторинга использования GPU:                                 \
    gom                                                                  \
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
    git+https://github.com/huggingface/transformers.git                  \
"
RUN ${PIP} install $PIPPARAMS --upgrade pip && \
    export TFVER=`${PIP} show tensorflow | grep Vers | awk '{print $2}'` && \
    ${PIP} install $PIPPARAMS tensorflow==${TFVER} $PYTHONMODULES --upgrade
# Установка всех необходимых пакетов при фиксации версии уже установленного Tensorflow.

#RUN pip install $PIPPARAMS $PYTHONMODULES

# Исправляем баг SAM2 с недоступностью конфигурационных файлов:
# RUN wget https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip && \
#     7z x ./main.zip && ls -laht && cd sam2-main && pip install -e ".[notebooks]" && cd .. && rm -rf sam2-main
#RUN pypackagesdir=`${PIP} show sam-2 | grep Location | cut -d' ' -f2` && \
RUN pypackagesdir=`${PIP} show sam2 | grep Location | cut -d' ' -f2` && \
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

# Грязно отключаем support_message, используемый в silence-tensorflow, т.к. он работает некорректно без root-a:
#RUN echo 'def support_message(*args, **kwargs): pass' > `python -c \
#    'import inspect, support_developer; print(inspect.getfile(support_developer.support_message))'`

# Создаём пользователя, даём ему право использовать sudo без пароля и устанавливаем его по умолчанию:
RUN useradd -ms "${SHELL_PATH}" "${USERNAME}"         && \
    adduser "${USERNAME}" sudo                        && \
    echo "${USERNAME}" ALL = NOPASSWD: ALL >> /etc/sudoers
USER "${USERNAME}"

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
RUN echo "c.LatexConfig.run_times = 2" >> "${JPCONF}"
# Двойной запуск компиляции LaTeX для проработки всяких ссылок.

#RUN export jupass=`python -c "from jupyter_server.auth import passwd; print(passwd('test'))"` && \
#    echo "c.NotebookApp.password = '${jupass}'"     >> "${JPCONF}"
    

# Поддержка LaTeX в JupyterLab:
#RUN jupyter lab --generate-config && \
#    echo "c.NotebookApp.extra_static_paths = ['node_modules/mathjax/es5/tex-mml-chtml.js']" >> ~/.jupyter/jupyter_lab_config.py

# Прописываем ряд полезных алиасов в "${BRC}":
#ARG JP="${HOMEDIR}"/.local/bin/jupyter-lab
ARG JP=jupyter-lab
ARG JPPARAMS="--notebook-dir '${WORKSPACE}'"
RUN echo 'alias jp="'"cd '${WORKSPACE}' && '${JP}' ${JPPARAMS} > /dev/null &"'"' >> "${BRC}" && \
    echo 'alias jl="'${JP}' list"'                                               >> "${BRC}" && \
    echo 'alias js="killall jupyter-lab"'                                        >> "${BRC}" && \
    echo 'alias ds="js"'                                                         >> "${BRC}" && \
    echo 'alias nv="nvidia-smi -l 1"'                                            >> "${BRC}" && \
    echo 'alias brc="nano '"${BRC}"' && source '"${BRC}"'"'                      >> "${BRC}" && \
    echo 'alias yt="yt-dlp"'                                                     >> "${BRC}" && \
    echo 'alias ffmpeg=/usr/bin/ffmpeg'                                          >> "${BRC}" && \
    echo 'alias ffprobe=/usr/bin/ffprobe'                                        >> "${BRC}" && \
    echo 'alias ffplay=/usr/bin/ffplay'                                          >> "${BRC}"


# Используемые порты:
EXPOSE 8888 6006 8000 8866
# JupyterLab, TensorBoard, KerasTuner, Voila.

# Точка монтирования внутри контейнера:
VOLUME  "${WORKSPACE}"
# Рабочая папка:
WORKDIR "${WORKSPACE}"

# Вносим рабочую дирректорию в список доверенных для Git:
RUN git config --global --add safe.directory "$WORKSPACE" && \
    git config --global --add safe.directory "$WORKSPACE"/3rdparty/dl_utils && \
    git config --global --add safe.directory "$WORKSPACE"/thirdparty/dl_utils && \
    git config --global core.fileMode false
#    git config --global user.email shubin.kit@ya.ru && \
#    git config --global user.name "Никита Шубин"

# Запуск Юпитер-сервера:
CMD JP="jupyter-lab"; \
    [[ -z "$JUPYTER_PORT" ]] || JP="$JP --port=$JUPYTER_PORT"; \
    [[ -z "$JUPYTER_PASS" ]] || JP="$JP --ServerApp.password="'"'"`echo ""$JUPYTER_PASS"" | python -c 'from jupyter_server.auth import passwd;print(passwd(input()))'`"'"'; \
    ${JP}
# Если в переменных окружения заданы пароль и порт, то они используются как параметры запуска.
