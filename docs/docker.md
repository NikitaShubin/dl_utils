# Docker-окружение

Контейнер с GPU, JupyterLab, OpenCode и всеми зависимостями `dl_utils`.
Корень основного проекта монтируется в `/workspace`, поэтому пути в коде
совпадают с путями на хосте.

## Запустить

```bash
docker/restart.sh
```

Скрипт останавливает старый контейнер, собирает образ и открывает консоль
от пользователя хоста. Для входа под root используйте `docker/openroot.sh`.
При первом запуске сборка занимает заметное время; при неудаче `run.sh`
пытается взять готовый образ `kitbrain/dl_cv:latest` с Docker Hub.

После запуска доступны:

- JupyterLab — `http://<host>:8888/`;
- OpenCode Web — `http://<host>:8000/`;
- фоновый OpenCode для TUI — порт `4097`.

TensorBoard и Voila автоматически не стартуют; их запускают вручную внутри
контейнера.

## Управление

```bash
docker/run.sh          # запуск
docker/stop.sh         # остановка
docker/open.sh         # консоль от пользователя хоста
docker/openroot.sh     # консоль от root
docker/restart.sh      # stop + run + open
```

Имя контейнера по умолчанию — `dl_cv_$USER`. Можно передать своё имя первым
аргументом:

```bash
docker/run.sh experiment_1
docker/open.sh experiment_1
docker/stop.sh experiment_1
```

`open.sh` ждёт до 90 секунд, пока `cmd.sh` подготовит окружение и напечатает
строку «Мониторинг». Если контейнер не успел — консоль всё равно откроется,
но возможны ошибки прав.

## Переменные окружения

Выставляйте их в `~/.zshenv` или `~/.profile` до запуска:

```bash
export JUPYTER_PASS='my_password'
export JUPYTER_PORT=8888
export OPENCODE_WEB_PASS='my_password'
export OPENCODE_WEB_PORT=8000
export OPENCODE_CLI_PORT=4097
export OLLAMA_HOST='http://localhost:11434'
```

- `JUPYTER_PASS` — пароль JupyterLab; без него вход без пароля.
- `JUPYTER_PORT` — порт JupyterLab.
- `OPENCODE_WEB_PASS` — пароль OpenCode Web; без него веб-сервер не
  запускается.
- `OPENCODE_WEB_PORT`, `OPENCODE_CLI_PORT` — порты OpenCode.
- `OLLAMA_HOST` — адрес Ollama; при заданном значении `ollm_utils`
  настраивает провайдеры для Jupyter-AI и OpenCode.

## Что монтируется

- корень основного проекта (`dl_utils/docker/../../../`) → `/workspace`;
- `~/projects/docker/ds/` → `/workspace/data/processed`;
- `/` → `/outroot`.

Путь к датасету задан в `docker/run.sh`; измените его под свой проект.
Монтирование `/` даёт контейнеру доступ ко всей файловой системе хоста,
поэтому не запускайте в нём недоверенный код.

## Как устроен запуск

`run.sh` собирает образ `kitbrain/dl_cv` с тегами `GIT-<hash>`, датой и
`latest`. Если сборка не удалась, берётся `latest` с Docker Hub. После
успешной сборки теги отправляются на Docker Hub в фоне.

Контейнер запускается с `--rm`, `--network=host`, `--ipc=host` и, если
доступен `nvidia-smi`, с `--runtime=nvidia --gpus all`. Внутри `cmd.sh`
подменяет UID/GID пользователя `user` на `LOCAL_UID`/`LOCAL_GID`, поэтому
файлы на хосте не становятся root-овыми.

## Ограничения

- `--network=host` и монтирование `/` рассчитаны на Linux; на macOS и Windows
  поведение будет другим.
- Порты берутся из окружения; при занятом порте `cmd.sh` завершает запуск с
  ошибкой. Проверьте `JUPYTER_PORT`, `OPENCODE_WEB_PORT` и
  `OPENCODE_CLI_PORT`.
- `OPENCODE_WEB_PASS` обязателен для веб-интерфейса OpenCode; без пароля
  сервис не поднимается.
- Контейнер удаляется при остановке (`--rm`), поэтому данные внутри него
  нужно хранить в смонтированных папках.
- Отправка образов на Docker Hub требует авторизации; ошибки отправки
  подавляются.
