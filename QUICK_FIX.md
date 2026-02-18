# ⚡ Быстрое решение проблемы с tiktoken

## Проблема
```
ERROR: Failed to build installable wheels for some pyproject.toml based projects (tiktoken)
```

## ✅ Решение (работает в 99% случаев)

Выполните эти команды по порядку:

```bash
# 1. Обновите pip
python -m pip install --upgrade pip

# 2. Установите tiktoken отдельно
python -m pip install tiktoken

# 3. Установите остальные зависимости
python -m pip install -r requirements.txt
```

## Почему это работает?

- Новые версии pip лучше находят предсобранные wheels (бинарные пакеты)
- Установка tiktoken отдельно гарантирует, что он установится первым
- После этого остальные пакеты установятся без проблем

## ✅ Проверка

После установки проверьте:

```bash
python -c "import tiktoken; print('✅ tiktoken работает!')"
python quick_start.py
```

## Если не помогло

См. подробные решения в [install_fix.md](install_fix.md)
