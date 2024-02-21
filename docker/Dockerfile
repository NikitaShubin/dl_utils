FROM tensorflow/tensorflow:latest-gpu

# Рабочая папка:
ARG WORKSPACE='/workspace'
#ENV WORKSPACE="${WORKSPACE}"

# Меняем sh на bash:
SHELL ["/bin/bash", "-l", "-c"]

# Чтобы apt не ждал ввода часового пояса:
#ENV DEBIAN_FRONTEND=noninteractive
# В данном случае не нужно, т.к. выбранный прообраз лишён этого недостатка.

# Устанавливаем доп. приложения:
RUN apt update &&                          \
    apt install -y --no-install-recommends \
           python3-pip                     \
           psmisc                          \
           curl                            \
           nethogs                         \
           htop                            \
           iotop                           \
           ffmpeg                          \
           nano                            \
           mc                              \
           git                             \
           graphviz                        \
           p7zip-full                      \
           pv                              \
           bsdmainutils                    \
           wget                            \
           yt-dlp                          \
           pandoc &&                       \
     apt upgrade -y &&                     \
     apt purge --autoremove &&             \
     apt autoclean &&                      \
     rm -rf /var/lib/apt/lists/*
# pandoc - для работы экспорта ipynb в HTML
# graphviz - для работы utils.plot_model в Keras
# nethogs - мониторинг загрузки сети
# curl и wget - консольные утилиты для загрузки файлов из сети
# yt-dlp - для загрузки видео/медиа-файлов с Youtube и не только
# iotop - мониторинг использования накопителей
# psmisc - для работы команды killall

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Устанавливаем необходимые python-библиотеки:
RUN pip install --upgrade pip &&                                 \
    pip install                                                  \
    treelib openpyxl                                             \
    albumentations                                               \
    ultralytics lapx                                             \
    git+https://github.com/facebookresearch/segment-anything.git \
    jupyterlab                                                   \
    onnx tf2onnx onnxmltools onnxconverter-common onnxruntime    \
    keras-tuner pydot                                            \
    silence-tensorflow                                           \
    --upgrade
#    keras keras-cv keras-tuner keras-nlp pydot                   \

#mathjax

# Поддержка LaTeX в JupyterLab:
#RUN jupyter lab --generate-config && \
#        echo "c.NotebookApp.extra_static_paths = ['node_modules/mathjax/es5/tex-mml-chtml.js']" >> ~/.jupyter/jupyter_lab_config.py

# Отключаем у JupyterLab вопрос об обновлении:
RUN jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

# Установка тёмной темы в JupyterLab:
ARG THEMEDIR=/root/.jupyter/lab/user-settings/\@jupyterlab/apputils-extension/
RUN mkdir -p "$THEMEDIR" && \
    echo '{ "theme":"JupyterLab Dark" }' > "$THEMEDIR"themes.jupyterlab-settings


# Прописываем ряд полезных алиасов в /root/.bashrc:
RUN echo 'alias jp="'"cd / && \
    nohup jupyter-lab \
    --no-browser \
    --allow-root \
    --NotebookApp.allow_origin='*' \
    --NotebookApp.ip='0.0.0.0' \
    --ip '*' \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.allow_remote_access=1 \
    --notebook-dir '${WORKSPACE}' > /dev/null &"'"' >> /root/.bashrc && \
    echo 'alias jl="jupyter-lab list"' >> /root/.bashrc && \
    #echo 'alias js="jupyter-lab stop"' >> /root/.bashrc && \
    echo 'alias js="killall jupyter-lab"' >> /root/.bashrc && \
    echo 'alias nv="nvidia-smi -l 1"' >> /root/.bashrc && \
    echo 'alias brc="nano ~/.bashrc && source ~/.bashrc"' >> /root/.bashrc && \
    echo 'alias yt="yt-dlp"' >> /root/.bashrc && \
    echo 'alias ffmpeg=/usr/bin/ffmpeg' >> /root/.bashrc


# Используемые порты:
EXPOSE 8888 6006
#EXPOSE map[8888/tcp:{}]
#EXPOSE map[6006/tcp:{}]

# Точка монтирования внутри контейнера:
VOLUME  "${WORKSPACE}"

# Рабочая папка:
WORKDIR "${WORKSPACE}"

#RUN source /root/.bashrc
#CMD ["/bin/bash", "-l", "-c", "jp"]
# Запуск Юпитер-сервера:
#CMD /root/.bashrc
#CMD ["bash", "-c", "source ~/.bashrc"]
#CMD jp
#CMD jupyter-lab --notebook-dir=/ --ip 0.0.0.0 --no-browser --allow-root
#CMD ["bash", "-c", "echo 1"]


#CMD ["sh -c jupyter-lab --no-browser --allow-root --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0' --ip '*' --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_remote_access=1 --notebook-dir /workspace"]
CMD jupyter-lab                         \
    --no-browser                        \
    --allow-root                        \
    --NotebookApp.allow_origin='*'      \
    --NotebookApp.ip='0.0.0.0'          \
    --ip '*'                            \
    --NotebookApp.token=''              \
    --NotebookApp.password=''           \
    --NotebookApp.allow_remote_access=1 \
    --notebook-dir "$WORKSPACE"
