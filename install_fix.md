# 🔧 Решение проблемы с установкой tiktoken

## Проблема
```
ERROR: Failed to build installable wheels for some pyproject.toml based projects (tiktoken)
```

## Решение 1: Обновление pip (самое простое) ⭐

Чаще всего проблема решается обновлением pip:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Почему это работает:** Новые версии pip лучше находят предсобранные wheels (бинарные пакеты) для вашей системы, избегая необходимости компиляции из исходников.

## Решение 2: Установка tiktoken отдельно

Если обновление pip не помогло, попробуйте установить tiktoken отдельно:

```bash
pip install tiktoken
pip install -r requirements.txt
```

## Решение 3: Установка Rust компилятора (если нужна сборка из исходников)

Если у вас старая система или специфичная архитектура, может потребоваться Rust:

### Windows:
1. Скачайте и установите [Rust](https://rustup.rs/)
2. Или установите через Visual Studio Build Tools

### Linux:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Mac:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

После установки Rust:
```bash
pip install -r requirements.txt
```

## Решение 4: Использование предсобранных wheels

Попробуйте установить с явным указанием использовать только wheels:

```bash
pip install --only-binary :all: -r requirements.txt
```

Если это не работает, попробуйте без этого флага (разрешить сборку из исходников после установки Rust).

## Решение 5: Альтернативная установка (пошагово)

Установите зависимости по одной, чтобы найти проблемную:

```bash
pip install --upgrade pip
pip install tiktoken
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install chromadb
pip install streamlit
pip install pypdf2
pip install python-dotenv
pip install openai
```

## Решение 6: Использование conda (если pip не работает)

Если у вас установлен Anaconda/Miniconda:

```bash
conda install -c conda-forge tiktoken
pip install -r requirements.txt
```

## Проверка установки

После установки проверьте:

```bash
python -c "import tiktoken; print('✅ tiktoken установлен успешно')"
python quick_start.py
```

## Если ничего не помогает

1. **Проверьте версию Python:** Должна быть 3.8 или выше
   ```bash
   python --version
   ```

2. **Используйте виртуальное окружение:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # или
   source venv/bin/activate  # Linux/Mac
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Попробуйте использовать HuggingFace вместо OpenAI** (не требует tiktoken для embeddings):
   - В `app.py` выберите "HuggingFace (бесплатно, локально)"
   - Но для генерации ответов все равно нужен OpenAI API ключ

## Дополнительная информация

- tiktoken используется OpenAI для подсчета токенов
- Проблема обычно возникает на старых системах или при использовании устаревшего pip
- В 99% случаев помогает простое обновление pip
