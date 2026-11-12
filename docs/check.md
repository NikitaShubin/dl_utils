# ✅ Проверка кода: check.sh

Комбинированная проверка Python и инфраструктуры: `check-py.sh` и
`check-infra.sh` подряд. Аргументы передаются обоим скриптам.

## 🚀 Примеры

```bash
# Проверить текущую папку
/path/to/dl_utils/check.sh

# Проверить конкретный проект
/path/to/dl_utils/check.sh ~/my_project

# Проверить отдельные файлы
./check.sh src module.py docker/Dockerfile README.md

# Автофиксы по правилам проекта
./check.sh -f -p .
```

`check.sh` — часть набора из трёх чекеров с единым стилем вызова. Общие
цели, режимы, флаги, тесты и коды возврата описаны в
[«Общих правилах чекеров»](check-common.md); здесь — только комбинирование
двух этапов.

## 🔧 Что происходит внутри

`check.sh` передаёт цели обоим подскриптам. Каждый берёт только свои типы
файлов:

- `.py`, `.ipynb` → `check-py.sh`;
- `Dockerfile`, `*.Dockerfile`, `docker-compose*.yml`,
  `docker-compose*.yaml`, `compose*.yml`, `compose*.yaml`, `*.sh`, `*.md` →
  `check-infra.sh`.

Этапы идут последовательно: сначала Python, затем инфраструктура. Падение
одного этапа не отменяет второй, но итоговый код будет `1`. Python-часть
выполняет [check-py.sh](check-py.md) (pytest с покрытием, ruff, mypy),
инфраструктурную — [check-infra.sh](check-infra.md) (Docker, shell и Markdown).

## ⚠️ Ограничения

- `check.sh` не умеет запускать только один этап. Для этого вызывайте
  `check-py.sh` или `check-infra.sh` напрямую.
