"""
Модуль для обработки PDF файлов и разбивки на чанки (chunks)

Что делает:
1. Загружает PDF файл
2. Извлекает текст с сохранением информации о страницах
3. Разбивает текст на небольшие кусочки (chunks) для лучшего поиска
"""

import PyPDF2
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessor:
    """
    Класс для обработки PDF файлов.
    
    Chunking (разбивка на кусочки) - важная часть RAG:
    - Слишком большие куски → плохой поиск (много лишнего)
    - Слишком маленькие → теряется контекст
    - Оптимальный размер: 500-1000 символов с перекрытием
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Инициализация процессора PDF.
        
        Args:
            chunk_size: Размер одного чанка в символах (по умолчанию 1000)
            chunk_overlap: Перекрытие между чанками (по умолчанию 200)
                          Перекрытие нужно, чтобы не терять контекст на границах
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Приоритет разделителей
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Извлекает текст из PDF с сохранением информации о страницах.
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Список словарей с текстом и номером страницы
            [{"text": "...", "page": 1}, ...]
        """
        chunks_with_metadata = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                print(f"📄 Загружаю PDF: {pdf_path}")
                print(f"📊 Всего страниц: {total_pages}")
                
                # Извлекаем текст с каждой страницы
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    text = page.extract_text()
                    
                    if text.strip():  # Пропускаем пустые страницы
                        chunks_with_metadata.append({
                            "text": text,
                            "page": page_num
                        })
                
                print(f"✅ Успешно извлечено {len(chunks_with_metadata)} страниц с текстом")
                return chunks_with_metadata
                
        except FileNotFoundError:
            print(f"❌ Ошибка: Файл {pdf_path} не найден")
            return []
        except Exception as e:
            print(f"❌ Ошибка при чтении PDF: {e}")
            return []
    
    def split_into_chunks(self, pages_data: List[Dict]) -> List[Dict]:
        """
        Разбивает текст на чанки с сохранением информации о страницах.
        
        Args:
            pages_data: Список словарей с текстом и номером страницы
            
        Returns:
            Список чанков с метаданными (текст, страница, индекс чанка)
        """
        all_chunks = []
        
        for page_data in pages_data:
            text = page_data["text"]
            page_num = page_data["page"]
            
            # Разбиваем текст страницы на чанки
            text_chunks = self.text_splitter.split_text(text)
            
            # Добавляем метаданные к каждому чанку
            for chunk_idx, chunk_text in enumerate(text_chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "metadata": {
                        "page": page_num,
                        "chunk_index": chunk_idx,
                        "source": f"page_{page_num}_chunk_{chunk_idx}"
                    }
                })
        
        print(f"📦 Создано {len(all_chunks)} чанков из {len(pages_data)} страниц")
        return all_chunks


# Пример использования
if __name__ == "__main__":
    # Тестирование модуля
    processor = PDFProcessor(chunk_size=500, chunk_overlap=100)
    
    # Замените на путь к вашему PDF файлу
    pdf_path = "example.pdf"
    
    # Извлекаем текст
    pages = processor.extract_text_from_pdf(pdf_path)
    
    if pages:
        # Разбиваем на чанки
        chunks = processor.split_into_chunks(pages)
        
        # Показываем первые 3 чанка
        print("\n" + "="*50)
        print("Примеры чанков:")
        print("="*50)
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\nЧанк {i} (Страница {chunk['page']}):")
            print(chunk['text'][:200] + "...")
