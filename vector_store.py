"""
Модуль для работы с векторной базой данных (ChromaDB)

Что делает:
1. Создает embeddings (векторные представления) текста
2. Сохраняет их в ChromaDB для быстрого поиска
3. Позволяет искать похожие фрагменты текста

Как работают embeddings:
- Текст → вектор чисел (например, 1536 чисел для OpenAI)
- Похожие тексты → похожие векторы
- Поиск = сравнение векторов (косинусное расстояние)
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStore:
    """
    Класс для работы с векторной базой данных.
    
    ChromaDB - простая локальная векторная БД:
    - Не требует установки сервера
    - Работает прямо в Python
    - Автоматически создает индексы для быстрого поиска
    """
    
    def __init__(self, 
                 collection_name: str = "lecture_notes",
                 use_openai: bool = True,
                 openai_api_key: Optional[str] = None,
                 persist_directory: str = "./chroma_db"):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: Название коллекции в ChromaDB
            use_openai: Использовать OpenAI embeddings (True) или HuggingFace (False)
            openai_api_key: API ключ OpenAI (если None, берется из .env)
            persist_directory: Директория для сохранения базы данных
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_openai = use_openai
        
        # Создаем директорию для БД, если её нет
        os.makedirs(persist_directory, exist_ok=True)
        
        # Инициализируем ChromaDB клиент
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Выбираем модель для embeddings
        if use_openai:
            # OpenAI embeddings (платно, но очень качественные)
            if not openai_api_key:
                from dotenv import load_dotenv
                load_dotenv()
                openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not openai_api_key:
                raise ValueError("Необходим OPENAI_API_KEY. Создайте файл .env с ключом.")
            
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            print("✅ Используются OpenAI embeddings")
        else:
            # HuggingFace embeddings (бесплатно, работает локально)
            # Требует: pip install sentence-transformers
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                print("✅ Используются HuggingFace embeddings (локально)")
            except Exception as e:
                if "sentence_transformers" in str(e) or "sentence-transformers" in str(e).lower():
                    raise ImportError(
                        "Для режима HuggingFace нужен пакет sentence-transformers. "
                        "Установите: pip install sentence-transformers"
                    ) from e
                raise
        
        # Создаем или получаем коллекцию
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Хранилище конспектов лекций"}
        )
        
        print(f"📚 Коллекция '{collection_name}' готова к работе")
    
    def add_documents(self, chunks: List[Dict]):
        """
        Добавляет документы (чанки) в векторную базу данных.
        
        Args:
            chunks: Список словарей с ключами:
                   - "text": текст чанка
                   - "page": номер страницы
                   - "metadata": дополнительные метаданные
        """
        if not chunks:
            print("⚠️ Нет документов для добавления")
            return
        
        texts = [chunk["text"] for chunk in chunks]
        metadatas = []
        ids = []
        
        # Подготавливаем метаданные и ID для каждого чанка
        for idx, chunk in enumerate(chunks):
            metadata = {
                "page": chunk.get("page", 0),
                "chunk_index": chunk.get("chunk_index", idx),
                "source": chunk.get("metadata", {}).get("source", f"chunk_{idx}")
            }
            metadatas.append(metadata)
            ids.append(f"chunk_{idx}_page_{metadata['page']}")
        
        # Создаем embeddings для всех текстов
        print(f"🔄 Создаю embeddings для {len(texts)} чанков...")
        embeddings_list = self.embeddings.embed_documents(texts)
        
        # Добавляем в ChromaDB
        self.collection.add(
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Добавлено {len(texts)} документов в векторную БД")
        print(f"📊 Всего документов в коллекции: {self.collection.count()}")
    
    def search_similar(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Ищет похожие фрагменты текста по запросу.
        
        Args:
            query: Текст запроса (вопрос пользователя)
            n_results: Количество результатов для возврата
            
        Returns:
            Список словарей с найденными фрагментами:
            [{
                "text": "...",
                "page": 5,
                "score": 0.85,  # степень похожести (0-1)
                "metadata": {...}
            }, ...]
        """
        if self.collection.count() == 0:
            print("⚠️ База данных пуста. Сначала добавьте документы.")
            return []
        
        # Создаем embedding для запроса
        query_embedding = self.embeddings.embed_query(query)
        
        # Ищем похожие документы
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Форматируем результаты
        similar_chunks = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                # Преобразуем расстояние в score (чем меньше расстояние, тем выше score)
                distance = results["distances"][0][i]
                score = 1 - distance  # Простое преобразование
                
                similar_chunks.append({
                    "text": results["documents"][0][i],
                    "page": results["metadatas"][0][i].get("page", 0),
                    "score": score,
                    "metadata": results["metadatas"][0][i]
                })
        
        return similar_chunks
    
    def clear_collection(self):
        """Очищает всю коллекцию."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Хранилище конспектов лекций"}
        )
        print("🗑️ Коллекция очищена")


# Пример использования
if __name__ == "__main__":
    pass  # Тестирование: см. example_usage.py или app.py
