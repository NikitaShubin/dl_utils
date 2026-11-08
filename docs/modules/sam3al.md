# sam3al.py — загрузка SAM 3.1

Пока модуль отвечает только за скачивание весов SAM 3.1 с ModelScope.
Инференс SAM3 живёт в [`PreAnnotation/SAM3.ipynb`](../../PreAnnotation/SAM3.ipynb)
и использует Ultralytics; обёртка в этом модуле появится позже.

**Зависимости:** `modelscope` (опционально), `clip`, `utils.py`.

## Скачать модель

```python
from sam3al import download_sam3

model_path = download_sam3('~/models')
print(model_path)  # /home/user/models/sam3.1_multiplex.pt
```

Если путь заканчивается на `.pt`, он считается именем файла. Иначе это папка,
в которую добавляется `sam3.1_multiplex.pt`. Существующий файл не
перекачивается.

Без аргумента модель кладётся в `~/models/sam3.1_multiplex.pt`.

## Запуск из консоли

```bash
python sam3al.py
```

Скачивает модель по умолчанию в `~/models`.

## Ограничения

- `modelscope` не входит в обязательные зависимости. Если пакета нет, а модели
  на диске тоже нет, `download_sam3` пробросит `ModuleNotFoundError`.
- Поддерживается только `sam3.1_multiplex.pt`; другое имя даст `ValueError`.
- Во время скачивания рядом создаётся временная папка `.temp_download`, она
  удаляется после перемещения файла.
- Модуль ещё не предоставляет инференс. Готовый сценарий — `SAM3.ipynb`; для
  разметки без SAM3 используйте `sam2al.py` или `samal.py`.
