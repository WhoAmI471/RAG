"""
Быстрый старт для тестирования RAG системы

Этот скрипт поможет быстро проверить, что все работает правильно.
"""

import os
import sys

# Для вывода в консоль Windows (cp1251) без ошибок на emoji
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env не будет загружен, ключ можно задать через переменные окружения

def check_dependencies():
    """Проверяет наличие всех необходимых библиотек."""
    print("🔍 Проверка зависимостей...")
    
    # (pip-пакет, имя модуля для import)
    required_packages = [
        ("langchain", "langchain"),
        ("langchain-openai", "langchain_openai"),
        ("langchain-community", "langchain_community"),
        ("chromadb", "chromadb"),
        ("streamlit", "streamlit"),
        ("pypdf2", "PyPDF2"),
        ("openai", "openai"),
        ("python-dotenv", "dotenv"),
    ]
    
    missing = []
    for pip_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name} - не установлен")
            missing.append(pip_name)
    
    if missing:
        print(f"\n⚠️ Установите недостающие пакеты:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("✅ Все зависимости установлены!\n")
    return True

def check_api_key():
    """Проверяет наличие API ключа OpenAI."""
    print("🔑 Проверка API ключа...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"  ✅ OPENAI_API_KEY найден (длина: {len(api_key)} символов)")
        return True
    else:
        print("  ⚠️ OPENAI_API_KEY не найден в .env файле")
        print("  💡 Создайте файл .env с содержимым:")
        print("     OPENAI_API_KEY=your_key_here")
        print("  💡 Или используйте HuggingFace embeddings (бесплатно)")
        return False

def main():
    print("="*60)
    print("🚀 Быстрая проверка RAG системы")
    print("="*60)
    print()
    
    # Проверка зависимостей
    deps_ok = check_dependencies()
    if not deps_ok:
        return
    
    # Проверка API ключа
    api_ok = check_api_key()
    
    print("="*60)
    print("📋 Следующие шаги:")
    print("="*60)
    print()
    print("1. Если API ключ не настроен:")
    print("   - Создайте файл .env с OPENAI_API_KEY")
    print("   - Или используйте HuggingFace в app.py (бесплатно)")
    print()
    print("2. Запустите веб-интерфейс:")
    print("   streamlit run app.py")
    print()
    print("3. Или используйте командную строку:")
    print("   python main.py --pdf your_file.pdf --question 'Ваш вопрос'")
    print()
    print("4. Для тестирования без API ключа:")
    print("   python main.py --pdf your_file.pdf --use-huggingface")
    print("   (но для генерации ответов все равно нужен OpenAI API ключ)")
    print()

if __name__ == "__main__":
    main()
