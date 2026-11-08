# Линтинг инфраструктуры: check-infra.sh

Проверка `docker-compose`, `Dockerfile`/`*.Dockerfile`, shell-скриптов и
Markdown. Конфиги `dclint`, `hadolint` и `markdownlint` берутся из
`dl_utils`, поэтому скрипт работает и в чужих проектах; `shellcheck` ищет
`.shellcheckrc` сам.

## Проверить проект

```bash
/path/to/dl_utils/check-infra.sh ~/my_project
```

Без путей проверяется текущая папка. Каждый файл проверяется из своей
директории; ошибка в одном файле не останавливает остальные.

## Проверить отдельные файлы

```bash
./check-infra.sh docker docs README.md
```

Поддерживаются:

- `docker-compose*.yml`, `docker-compose*.yaml`, `compose*.yml`,
  `compose*.yaml` — `dclint`;
- `Dockerfile`, `*.Dockerfile` — `hadolint`;
- `*.sh` — `shellcheck`;
- `*.md` — `markdownlint`.

Файлы других типов игнорируются.

## Флаги

- `-f`, `--fix` — автофиксы `dclint` и `markdownlint`. `hadolint` и
  `shellcheck` только отчитываются.
- `-g`, `--git-only` — только закоммиченные файлы.
- `-c`, `--clean-cache` — принимается для совместимости с `check.sh`, на
  infra-проверки не влияет.
- `-q`, `--quiet-no-files` — не печатать сообщение об отсутствии файлов,
  вернуть код `3`.
- `-H`, `--print-header` — печатать шапку с режимом и целями.

## Что происходит внутри

Файлы ищутся по диску или берутся из `git ls-files`. Служебные каталоги
(`.git`, `__pycache__`, `.venv`, `node_modules`, `dist`, `build`, кеши)
пропускаются. Для каждой категории запускается свой линтер; файлы
перечисляются по одному, диагностика линтера выводится сразу после имени
файла.

Конфиги `dclint`, `hadolint` и `markdownlint` всегда берутся из корня
`dl_utils`. `check-infra` не передаёт `shellcheck` отдельный конфиг: он ищет
`.shellcheckrc` вверх по дереву от проверяемого файла. В корне `dl_utils`
такой файл есть, но для целей вне репозитория могут действовать правила их
проекта.

## Ограничения

- Нужны установленные `dclint`, `hadolint`, `shellcheck` и `markdownlint`.
- `hadolint` и `shellcheck` не умеют автофиксы; `-f` их не меняет.
- `markdownlint` проверяет длину строк (88 символов) и другие правила из
  `.markdownlint.yaml`; длинные ссылки и таблицы лучше переносить.
- Вне git-репозитория `-g` не работает, а обычный поиск может занять время
  на большой файловой системе.
