Привет!

У меня есть репозиторий https://github.com/NikitaShubin/dl_utils

Полностью перепиши его README.md на русском языке с учётом содержания репозитория. Дай краткое описание всем модулям. Не нумеруй их, но можешь разбить на группы. В заголовках используй значки. Среди групп должны быть такие: глубокое обучение, авторазметка, работа с CVAT, утилиты общего назначения и т.п. По возможности название модулей указывай со ссылкой на соответствующий документ, если он есть в https://github.com/NikitaShubin/dl_utils/tree/main/docs, но ссылка должна быть относительной (например ./docs/utils.md).

После перечисления основных модулей с их кратким описанием добавь под спойлер mermaid-диаграмму, где было бы дерево зависимостей всех модулей репозитория друг от друга слева на право. Сами зависимости прочитай в содержимом описываемых файлов. Bash-файлов в графе быть не должно. Удаляй все прямые зависимости там, где уже есть транзитивные. Т.е. из множества путей между двумя узлами графа оставлять надо только самые длинные! Например, если имеются две цепочки зависимостей A -> C и A -> B -> C, то рисовать нужно только A -> B -> C. Можешь пожертвовать логикой разбиения модулей на уровни, если это поможет сделать граф более читаемым. Но ветвление дерева графа должно начинаться с utils.py, слева направо, т.к это единственный модуль, не использующий ни один другой в репозитории.

Потом также перечисли все bash-скрипты.

Продолжи небольшим описанием докера, в который включены все зависимости проекта. Сам репозиторий dl_utils должен включаться в другие проекты как подмодуль и распологаться в папке главный_проект/3rdparty_(или_аналогичная)/dl_utils. Этой логикой продиктован способ использования контейнера, который мапит в свою папку /workspace папку главного проекта, находящуюся двумя уровнями выше репозитория dl_utils.

Способ установки проета не описывай.

Далее должно быть краткое описание dockerPreAnnotation для интерактивной сегментации видео на базе SAM2

В заключении обозначь имя автора.

Для просмотра содержимого основных файлов репозитория пройди по следующим ссылкам:

https://github.com/NikitaShubin/dl_utils/blob/main/LICENSE
https://github.com/NikitaShubin/dl_utils/blob/main/.gitignore
https://github.com/NikitaShubin/dl_utils/blob/main/README.md

https://github.com/NikitaShubin/dl_utils/blob/main/utils.py
https://github.com/NikitaShubin/dl_utils/blob/main/video_utils.py
https://github.com/NikitaShubin/dl_utils/blob/main/alb_utils.py
https://github.com/NikitaShubin/dl_utils/blob/main/classes.py
https://github.com/NikitaShubin/dl_utils/blob/main/copybal.py
https://github.com/NikitaShubin/dl_utils/blob/main/cv_utils.py
https://github.com/NikitaShubin/dl_utils/blob/main/cvat.py
https://github.com/NikitaShubin/dl_utils/blob/main/cvat_srv.py
https://github.com/NikitaShubin/dl_utils/blob/main/dinoal.py
https://github.com/NikitaShubin/dl_utils/blob/main/ipy_utils.py
https://github.com/NikitaShubin/dl_utils/blob/main/keras_utils.py
https://github.com/NikitaShubin/dl_utils/blob/main/ml_utils.py
https://github.com/NikitaShubin/dl_utils/blob/main/onnx_utils.py
https://github.com/NikitaShubin/dl_utils/blob/main/pt_utils.py
https://github.com/NikitaShubin/dl_utils/blob/main/sam2al.py
https://github.com/NikitaShubin/dl_utils/blob/main/samal.py
https://github.com/NikitaShubin/dl_utils/blob/main/seg.py
https://github.com/NikitaShubin/dl_utils/blob/main/tf_utils.py
https://github.com/NikitaShubin/dl_utils/blob/main/tfmot_utils.py
https://github.com/NikitaShubin/dl_utils/blob/main/yolo.py

Отдельно рассмотри и опиши назначение bash-скриптов:
https://github.com/NikitaShubin/dl_utils/blob/main/rerun.sh
https://github.com/NikitaShubin/dl_utils/blob/main/run.sh
https://github.com/NikitaShubin/dl_utils/blob/main/show.sh
https://github.com/NikitaShubin/dl_utils/blob/main/split-video.sh

В репозиторий включён докер со всеми необходимыми зависимостями:
https://github.com/NikitaShubin/dl_utils/blob/main/docker/Dockerfile
https://github.com/NikitaShubin/dl_utils/blob/main/docker/run.sh
https://github.com/NikitaShubin/dl_utils/blob/main/docker/stop.sh

Особняком стоит инструмент предразметки (о нём тоже скажи пару слов):
https://github.com/NikitaShubin/dl_utils/blob/main/dockerPreAnnotation/Dockerfile
https://github.com/NikitaShubin/dl_utils/blob/main/dockerPreAnnotation/PreAnnotation.ipynb