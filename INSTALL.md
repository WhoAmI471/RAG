# 🚀 Инструкция по установке

## Шаг 1: Установка Python

Убедитесь, что у вас установлен Python 3.8 или выше:

```bash
python --version
```

Если Python не установлен, скачайте с [python.org](https://www.python.org/downloads/)

## Шаг 2: Установка зависимостей

**ВАЖНО:** Сначала обновите pip, чтобы избежать проблем с tiktoken:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Примечание:** Если используете виртуальное окружение (рекомендуется):

```bash
# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Активация (Linux/Mac)
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

## Шаг 3: Настройка API ключа

### Вариант A: OpenAI (платно, но качественно)

1. Получите API ключ на [platform.openai.com](https://platform.openai.com/api-keys)
2. Создайте файл `.env` в корне проекта:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Вариант B: HuggingFace (бесплатно, локально)

1. Не нужен API ключ для embeddings
2. Первый запуск скачает модель (~400MB)
3. Работает полностью локально
4. **НО:** Для генерации ответов все равно нужен OpenAI API ключ

## Шаг 4: Проверка установки

Запустите скрипт проверки:

```bash
python quick_start.py
```

Он проверит:
- ✅ Все ли библиотеки установлены
- ✅ Настроен ли API ключ
- ✅ Даст инструкции по следующим шагам

## Шаг 5: Запуск

### Веб-интерфейс (рекомендуется):

```bash
streamlit run app.py
```

Откроется браузер с интерфейсом.

### Командная строка:

```bash
python main.py --pdf your_file.pdf --question "Ваш вопрос"
```

## 🐛 Решение проблем

### Ошибка: "pip не найден"

**Решение:** Используйте `python -m pip` вместо `pip`:
```bash
python -m pip install -r requirements.txt
```

### Ошибка при установке ChromaDB

**Решение:** Убедитесь, что у вас установлены компиляторы C++:
- Windows: [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
- Linux: `sudo apt-get install build-essential`
- Mac: `xcode-select --install`

### Ошибка: "OPENAI_API_KEY не найден"

**Решение:** 
1. Убедитесь, что файл `.env` создан в корне проекта
2. Проверьте, что ключ указан правильно (без пробелов, кавычек)
3. Или используйте HuggingFace embeddings

### Ошибка: "Failed to build installable wheels for tiktoken"

**Решение:** Это самая частая проблема. См. подробные решения в [install_fix.md](install_fix.md)

**Быстрое решение:**
```bash
# 1. Обновите pip
python -m pip install --upgrade pip

# 2. Установите tiktoken отдельно
pip install tiktoken

# 3. Затем установите остальные зависимости
pip install -r requirements.txt
```

Если не помогло, установите Rust компилятор или используйте альтернативные методы из `install_fix.md`

### Медленная установка

**Решение:** Используйте зеркало pip:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 📚 Дополнительная информация

После установки прочитайте [README.md](README.md) для подробного объяснения работы системы.
