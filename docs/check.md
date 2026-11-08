# Проверка кода: check.sh

Комбинированная проверка Python и инфраструктуры: `check-py.sh` и
`check-infra.sh` подряд. Конфиги `ruff`, `mypy`, `dclint`, `hadolint` и
`markdownlint` берутся из `dl_utils`, поэтому скрипт работает и в чужих
проектах; `shellcheck` ищет `.shellcheckrc` вверх по дереву от файла.

## Проверить свой проект

```bash
/path/to/dl_utils/check.sh
```

Без путей из любой папки, кроме корня `dl_utils`, проверяется эта папка.
Можно передать путь явно:

```bash
/path/to/dl_utils/check.sh ~/my_project
```

## Проверить отдельные файлы

```bash
./check.sh src module.py docker/Dockerfile README.md
```

`check.sh` передаёт цели обоим подскриптам. Каждый берёт только свои типы
файлов:

- `.py`, `.ipynb` → `check-py.sh`;
- `Dockerfile`, `*.Dockerfile`, `docker-compose*.yml`,
  `docker-compose*.yaml`, `compose*.yml`, `compose*.yaml`, `*.sh`, `*.md` →
  `check-infra.sh`.

Если ни один подскрипт не нашёл подходящих файлов, выводится предупреждение
и код `0`.

## Флаги

```bash
./check.sh -f src      # автоисправления
./check.sh -g .        # только закоммиченное в git
./check.sh -c .        # очистить кеши инструментов
```

- `-f`, `--fix` — разрешить автофиксы (ruff format/check, dclint,
  markdownlint). По умолчанию файлы не изменяются.
- `-g`, `--git-only` — проверять только файлы из `git ls-files`.
- `-c`, `--clean-cache` — удалить кеши mypy/ruff/pytest перед прогоном.
  На infra-проверки не влияет.

## Результат

- `0` — ошибок нет;
- `1` — хотя бы один чекер нашёл проблемы или путь не существует;
- `2` — неизвестный флаг.

Этапы идут последовательно: сначала Python, затем инфраструктура. Падение
одного этапа не отменяет второй, но итоговый код будет `1`. Python-часть
выполняет [check-py.sh](check-py.md) (pytest с покрытием, ruff, mypy),
инфраструктурную — [check-infra.sh](check-infra.md) (Docker, shell и Markdown).

## Ограничения

- Для проверки нужны сами инструменты: `ruff`, `mypy`, `pytest`, `dclint`,
  `hadolint`, `shellcheck`, `markdownlint`. В минимальном окружении они
  отсутствуют — используйте Docker-образ.
- `-g` требует git-репозиторий; цель вне репозитория считается ошибкой.
- Скрипт не устанавливает зависимости проверяемого проекта. Если проект не
  импортируется, mypy и pytest упадут.
