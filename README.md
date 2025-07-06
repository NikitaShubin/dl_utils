# 🧠 dl_utils: Утилиты для глубокого обучения

Набор инструментов для задач технического зрения, работы с разметкой и автоматизации процессов глубокого обучения. Репозиторий предназначен для подключения в основной проект как подмодуль.

## 🧩 Основные модули

### 🛠️ Утилиты общего назначения
- **[utils.py](./docs/utils.md "Перейти к документации")**: Базовые функции (работа с файлами, логирование, измерения времени)
- **[ipy_utils.py](./docs/ipy_utils.md "Перейти к документации")**: Инструменты для Jupyter Notebooks

### 🎥 Работа с видео
- **[video_utils.py](./docs/video_utils.md "Перейти к документации")**: Захват, обработка и сохранение видео
- **[split-video.sh](./docs/split_video.md "Перейти к документации")**: Скрипт для разделения видео на фрагменты

### 🖼️ Техническое зрение (CV)
- **[cv_utils.py](./docs/cv_utils.md "Перейти к документации")**: Обработка изображений, визуализация аннотаций
- **[alb_utils.py](./docs/alb_utils.md "Перейти к документации")**: Аугментации на базе Albumentations

### ⚙️ Авторазметка
- **[samal.py](./docs/samal.md "Перейти к документации")**: Автоматическая сегментация изображений с [SAM](https://github.com/facebookresearch/segment-anything "Перейти к репозиторию")
- **[sam2al.py](./docs/sam2al.md "Перейти к документации")**: Автоматическая сегментация видео с [SAM2](https://github.com/facebookresearch/sam2 "Перейти к репозиторию")
- **[dinoal.py](./docs/dinoal.md "Перейти к документации")**: Автоматическая детекция объектов с [DINO](https://github.com/IDEA-Research/GroundingDINO "Перейти к репозиторию")

### 🏷️ Работа с CVAT
- **[cvat.py](./docs/cvat.md "Перейти к документации")**: API клиент для CVAT
- **[cvat_srv.py](./docs/cvat_srv.md "Перейти к документации")**: Утилиты для работы серверных задач CVAT
- **[copybal.py](./docs/copybal.md "Перейти к документации")**: Балансировка датасетов между проектами CVAT
- **[classes.py](./docs/classes.md "Перейти к документации")**: Работа с метками, используя таблицы [классов](classes_template.xlsx "перейти к примеру файла") и [суперклассов](superclasses_template.xlsx "перейти к примеру файла")

### 🤖 Глубокое обучение
- **[pt_utils.py](./docs/pt_utils.md "Перейти к документации")**: Утилиты для PyTorch (работа с моделями, данными)
- **[tf_utils.py](./docs/tf_utils.md "Перейти к документации")**: Инструменты для TensorFlow/Keras
- **[tfmot_utils.py](./docs/tfmot_utils.md "Перейти к документации")**: Квантование моделей TensorFlow
- **[ml_utils.py](./docs/ml_utils.md "Перейти к документации")**: Общие ML функции (метрики, обработка данных)
- **[onnx_utils.py](./docs/onnx_utils.md "Перейти к документации")**: Конвертация и работа с ONNX-моделями
- **[yolo.py](./docs/yolo.md "Перейти к документации")**: Утилиты для работы с YOLO-моделями
- **[seg.py](./docs/seg.md "Перейти к документации")**: Сегментация изображений

<details>
<summary>🔄 Зависимости модулей</summary>

```mermaid
graph RL;
    node_0[pt_utils];
    node_1[copybal];
    node_2[seg];
    node_3[sam2al];
    node_4[alb_utils];
    node_5[ipy_utils];
    node_6[tf_utils];
    node_7[video_utils];
    node_8[cvat_srv];
    node_9[cvat];
    node_10[yolo];
    node_11[onnx_utils];
    node_12[cv_utils];
    node_13[samal];
    node_14[tfmot_utils];
    node_15[keras_utils];
    node_17[classes];
    node_16[dinoal];
    node_18[utils];
    node_19[ml_utils];
    %% Выравнивание стоков на одном уровне
    subgraph SinkGroup [ ]
        direction LR
        node_0
        node_18
    end
    style SinkGroup fill:none,stroke:none;
    node_1 --> node_18;
    node_2 --> node_9;
    node_3 --> node_13;
    node_4 --> node_18;
    node_5 --> node_9;
    node_6 --> node_4;
    node_7 --> node_18;
    node_8 --> node_9;
    node_9 --> node_7;
    node_9 --> node_12;
    node_10 --> node_1;
    node_10 --> node_19;
    node_10 --> node_9;
    node_11 --> node_19;
    node_12 --> node_18;
    node_13 --> node_0;
    node_13 --> node_9;
    node_14 --> node_15;
    node_15 --> node_18;
    node_16 --> node_0;
    node_16 --> node_9;
    node_17 --> node_18;
    node_19 --> node_18;
```
</details>

## `>_` Bash-скрипты
- **rerun.sh**: Перезапуск скриптов при изменении кода
- **show.sh**: Вывод последних строк локфайлов в реальном времени
- **split-video.sh**: Разделение видео на фрагменты по времени

## 🐳 Docker окружение
Проект включает Docker-контейнер со всеми зависимостями.

**Скрипты управления контейнером:**
- `docker/run.sh`: Запуск контейнера с поддержкой GPU
- `docker/stop.sh`: Остановка контейнера

**Особенности использования:**
1. Репозиторий подключается как подмодуль в `main_project/*/dl_utils`, где `*` - папка для подмодулей (например, `3rdparty`)
2. Контейнер монтирует корень главного проекта в `/workspace`
3. Пути в коде рассчитываются относительно расположения `dl_utils`

## ✨ Интерактивная предразметка (PreAnnotation)
Отдельный инструмент для автоматической детекции (GroundingDINO) или интерактивной сегментации (SAM2) объектов на  фото/видео:
- Ноутбуки для управления процессом разметки
- Поддержка GPU-ускорения
- Автоматизированная генерация масок/обрамляющих прямоугольников объектов
- Выгрузка результатов в CVAT (через API)

**Основные компоненты:**
- `dockerPreAnnotation/Dockerfile`: Сборка образа
- `run.sh`/`stop.sh`/`drs.sh`: Запуск/остановака/перезапуск докера

---
👨‍💻Автор: **[Никита](https://disk.yandex.ru/i/2HfPHtSlAJJuyQ "см. резюме") [Шубин](https://disk.yandex.ru/i/BxSVPalOlTq4GA "my Curriculum vitae")**  
Лицензия: [MIT](./LICENSE)
