"""
Пример использования RAG системы

Этот файл показывает, как использовать все компоненты системы пошагово.
"""

from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_chain import RAGChain
import os
from dotenv import load_dotenv

load_dotenv()


def example_basic_usage():
    """
    Базовый пример использования RAG системы.
    
    Шаг за шагом показывает весь процесс:
    1. Загрузка PDF
    2. Разбивка на чанки
    3. Создание embeddings
    4. Поиск похожих фрагментов
    5. Генерация ответа
    """
    
    print("="*60)
    print("📚 Пример использования RAG системы")
    print("="*60)
    print()
    
    # ============================================================
    # ШАГ 1: Обработка PDF файла
    # ============================================================
    print("ШАГ 1: Обработка PDF файла")
    print("-"*60)
    
    pdf_path = "example.pdf"  # Замените на путь к вашему PDF
    
    if not os.path.exists(pdf_path):
        print(f"⚠️ Файл {pdf_path} не найден.")
        print("   Создайте тестовый PDF или укажите путь к существующему файлу.")
        return
    
    # Создаем процессор PDF
    processor = PDFProcessor(
        chunk_size=1000,      # Размер чанка: 1000 символов
        chunk_overlap=200     # Перекрытие: 200 символов
    )
    
    # Извлекаем текст из PDF
    pages = processor.extract_text_from_pdf(pdf_path)
    
    if not pages:
        print("❌ Не удалось извлечь текст из PDF")
        return
    
    # Разбиваем на чанки
    chunks = processor.split_into_chunks(pages)
    print(f"✅ Создано {len(chunks)} чанков из {len(pages)} страниц\n")
    
    # ============================================================
    # ШАГ 2: Создание векторного хранилища
    # ============================================================
    print("ШАГ 2: Создание векторных представлений (embeddings)")
    print("-"*60)
    
    # Выбираем модель для embeddings
    use_openai = True  # False для использования HuggingFace (бесплатно)
    
    if use_openai and not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY не найден. Используем HuggingFace...")
        use_openai = False
    
    # Создаем векторное хранилище
    vector_store = VectorStore(
        collection_name="lecture_notes",
        use_openai=use_openai
    )
    
    # Добавляем чанки в базу данных
    vector_store.add_documents(chunks)
    print(f"✅ Документы добавлены в векторную БД\n")
    
    # ============================================================
    # ШАГ 3: Поиск похожих фрагментов
    # ============================================================
    print("ШАГ 3: Поиск похожих фрагментов")
    print("-"*60)
    
    # Пример вопроса
    question = "Что такое машинное обучение?"
    print(f"Вопрос: {question}\n")
    
    # Ищем похожие фрагменты
    similar_chunks = vector_store.search_similar(question, n_results=3)
    
    print(f"Найдено {len(similar_chunks)} похожих фрагментов:\n")
    for i, chunk in enumerate(similar_chunks, 1):
        print(f"{i}. Страница {chunk['page']} (релевантность: {chunk['score']:.2f})")
        print(f"   {chunk['text'][:150]}...")
        print()
    
    # ============================================================
    # ШАГ 4: Генерация ответа через LLM
    # ============================================================
    print("ШАГ 4: Генерация ответа через LLM")
    print("-"*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY необходим для генерации ответов")
        print("   Создайте файл .env с вашим API ключом")
        return
    
    # Создаем RAG цепочку
    rag_chain = RAGChain(
        vector_store=vector_store,
        model_name="gpt-3.5-turbo",  # или "gpt-4"
        temperature=0.7
    )
    
    # Генерируем ответ
    result = rag_chain.generate_answer(question, n_chunks=3)
    
    print(f"\n📝 Ответ:")
    print("-"*60)
    print(result["answer"])
    print()
    
    print("📚 Источники:")
    print("-"*60)
    for i, source in enumerate(result["sources"], 1):
        print(f"{i}. Страница {source['page']} (релевантность: {source['score']:.2f})")
        print(f"   {source['text'][:150]}...")
        print()
    
    print("="*60)
    print("✅ Пример завершен!")
    print("="*60)


def example_multiple_questions():
    """
    Пример работы с несколькими вопросами подряд.
    """
    print("\n" + "="*60)
    print("💬 Пример: Несколько вопросов подряд")
    print("="*60)
    
    # Здесь можно загрузить базу один раз и задавать много вопросов
    # (код аналогичен example_basic_usage, но без повторной загрузки)
    
    print("💡 Идея: Загрузите базу один раз, затем задавайте много вопросов!")
    print("   Это экономит время и API запросы.")


if __name__ == "__main__":
    # Запускаем базовый пример
    example_basic_usage()
    
    # Показываем пример с несколькими вопросами
    example_multiple_questions()
