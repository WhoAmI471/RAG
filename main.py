"""
Главный скрипт для быстрого тестирования RAG системы без веб-интерфейса

Использование:
    python main.py --pdf path/to/file.pdf --question "Ваш вопрос"
"""

import argparse
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_chain import RAGChain
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="RAG чат-бот по конспектам")
    parser.add_argument("--pdf", type=str, help="Путь к PDF файлу")
    parser.add_argument("--question", type=str, help="Вопрос для бота")
    parser.add_argument("--use-huggingface", action="store_true", 
                       help="Использовать HuggingFace вместо OpenAI для embeddings")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                       help="Модель LLM (gpt-3.5-turbo или gpt-4)")
    
    args = parser.parse_args()
    
    # Проверяем наличие PDF файла
    if not args.pdf or not os.path.exists(args.pdf):
        print("❌ Укажите путь к существующему PDF файлу")
        print("Пример: python main.py --pdf lecture.pdf --question 'Что такое RAG?'")
        return
    
    print("="*60)
    print("🚀 Запуск RAG системы")
    print("="*60)
    
    # Шаг 1: Обработка PDF
    print("\n📄 Шаг 1: Обработка PDF файла...")
    processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    pages = processor.extract_text_from_pdf(args.pdf)
    
    if not pages:
        print("❌ Не удалось извлечь текст из PDF")
        return
    
    chunks = processor.split_into_chunks(pages)
    print(f"✅ Создано {len(chunks)} чанков")
    
    # Шаг 2: Создание векторного хранилища
    print("\n🔢 Шаг 2: Создание векторных представлений...")
    use_openai = not args.use_huggingface
    
    if use_openai and not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY не найден. Используйте --use-huggingface для бесплатного варианта")
        return
    
    vector_store = VectorStore(
        collection_name="lecture_notes",
        use_openai=use_openai
    )
    vector_store.add_documents(chunks)
    
    # Шаг 3: Создание RAG цепочки
    print("\n🤖 Шаг 3: Инициализация LLM...")
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY необходим для генерации ответов")
        return
    
    rag_chain = RAGChain(
        vector_store=vector_store,
        model_name=args.model
    )
    
    # Шаг 4: Обработка вопроса (если указан)
    if args.question:
        print(f"\n💬 Шаг 4: Обработка вопроса: '{args.question}'")
        print("-"*60)
        
        result = rag_chain.generate_answer(args.question, n_chunks=3)
        
        print("\n📝 Ответ:")
        print(result["answer"])
        
        print("\n📚 Источники:")
        for i, source in enumerate(result["sources"], 1):
            print(f"\n{i}. Страница {source['page']} (релевантность: {source['score']:.2f})")
            print(f"   {source['text'][:150]}...")
    else:
        print("\n✅ Система готова к работе!")
        print("Запустите Streamlit приложение: streamlit run app.py")
        print("Или задайте вопрос через командную строку:")
        print(f"python main.py --pdf {args.pdf} --question 'Ваш вопрос'")


if __name__ == "__main__":
    main()
