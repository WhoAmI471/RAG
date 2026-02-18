"""
Модуль для генерации ответов на основе найденных фрагментов (RAG Chain)

Что делает:
1. Получает вопрос пользователя
2. Находит похожие фрагменты в базе данных
3. Отправляет их в LLM (ChatGPT/Claude) с инструкцией ответить на вопрос
4. Возвращает ответ с ссылками на страницы

Это и есть "Retrieval-Augmented Generation":
- Retrieval = поиск релевантных фрагментов
- Augmented = дополнение промпта найденными данными
- Generation = генерация ответа LLM
"""

from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Провайдеры LLM (OpenAI, DeepSeek, Qwen)
def _get_openai_llm(model_name: str, temperature: float, api_key: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=api_key,
    )


def _get_deepseek_llm(temperature: float, api_key: str):
    """DeepSeek — бесплатные лимиты, API совместим с OpenAI. Ключ: platform.deepseek.com"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
    )


def _get_qwen_llm(temperature: float, api_key: str):
    """Qwen (Alibaba DashScope) — бесплатный tier. Ключ: dashscope.aliyun.com"""
    try:
        from langchain_qwq import ChatQwen
    except ImportError:
        raise ImportError(
            "Для Qwen установите: pip install langchain-qwq. "
            "Ключ API: DASHSCOPE_API_KEY (dashscope.aliyun.com)"
        )
    os.environ["DASHSCOPE_API_KEY"] = api_key
    return ChatQwen(
        model="qwen-turbo",
        temperature=temperature,
        max_tokens=2048,
    )


# Список доступных провайдеров и переменных окружения
LLM_PROVIDERS = {
    "openai": ("OPENAI_API_KEY", _get_openai_llm, "gpt-3.5-turbo"),
    "deepseek": ("DEEPSEEK_API_KEY", lambda t, k: _get_deepseek_llm(t, k), "deepseek-chat"),
    "qwen": ("DASHSCOPE_API_KEY", lambda t, k: _get_qwen_llm(t, k), "qwen-turbo"),
}


class RAGChain:
    """
    Класс для генерации ответов через RAG.
    
    Как это работает:
    1. Вопрос → поиск похожих фрагментов
    2. Фрагменты + вопрос → промпт для LLM
    3. LLM генерирует ответ на основе найденных данных
    """
    
    def __init__(self, 
                 vector_store,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 openai_api_key: Optional[str] = None,
                 use_llm: bool = True,
                 llm_provider: str = "openai"):
        """
        Инициализация RAG цепочки.
        
        Args:
            vector_store: Экземпляр VectorStore для поиска
            model_name: Название модели (для OpenAI: gpt-3.5-turbo/gpt-4; для deepseek/qwen не используется)
            temperature: Креативность ответов (0-1)
            openai_api_key: API ключ (для OpenAI). Для DeepSeek/Qwen ключ берётся из .env по провайдеру
            use_llm: Если False — только поиск фрагментов, без вызова LLM
            llm_provider: "openai" | "deepseek" | "qwen"
        """
        self.vector_store = vector_store
        self.use_llm = use_llm
        self.llm = None
        load_dotenv()
        
        if use_llm:
            if llm_provider not in LLM_PROVIDERS:
                raise ValueError(f"Неизвестный провайдер: {llm_provider}. Доступны: {list(LLM_PROVIDERS)}")
            env_var, get_llm, default_model = LLM_PROVIDERS[llm_provider]
            api_key = openai_api_key or os.getenv(env_var)
            if not api_key:
                raise ValueError(
                    f"Необходим {env_var} в файле .env для провайдера {llm_provider}. "
                    "Или выберите режим «Только поиск»."
                )
            from langchain_core.messages import HumanMessage, SystemMessage
            self._HumanMessage = HumanMessage
            self._SystemMessage = SystemMessage
            try:
                if llm_provider == "openai":
                    self.llm = get_llm(model_name, temperature, api_key)
                else:
                    self.llm = get_llm(temperature, api_key)
                print(f"✅ RAG цепочка инициализирована ({llm_provider}, модель: {model_name if llm_provider == 'openai' else default_model})")
            except Exception as e:
                error_msg = str(e)
                if "model" in error_msg.lower() and ("not found" in error_msg.lower() or "404" in error_msg or "does not exist" in error_msg.lower()):
                    raise ValueError(
                        f"Модель «{model_name if llm_provider == 'openai' else default_model}» недоступна. "
                        f"Проверьте доступность модели в вашем аккаунте или выберите другую модель."
                    ) from e
                raise
        else:
            print("✅ RAG цепочка инициализирована (режим «Только поиск», без LLM)")
    
    def generate_answer(self, question: str, n_chunks: int = 3) -> Dict:
        """
        Генерирует ответ на вопрос пользователя.
        
        Args:
            question: Вопрос пользователя
            n_chunks: Количество похожих фрагментов для использования
            
        Returns:
            Словарь с ответом и метаданными:
            {
                "answer": "Ответ на вопрос...",
                "sources": [
                    {"page": 5, "text": "...", "score": 0.85},
                    ...
                ],
                "question": "Исходный вопрос"
            }
        """
        # Шаг 1: Поиск похожих фрагментов
        print(f"🔍 Ищу похожие фрагменты для вопроса: '{question[:50]}...'")
        similar_chunks = self.vector_store.search_similar(question, n_results=n_chunks)
        
        if not similar_chunks:
            return {
                "answer": "Извините, не удалось найти релевантную информацию в базе данных. Убедитесь, что документы были загружены.",
                "sources": [],
                "question": question
            }
        
        # Шаг 2: Формируем контекст из найденных фрагментов
        context_parts = []
        for i, chunk in enumerate(similar_chunks, 1):
            context_parts.append(
                f"[Фрагмент {i}, Страница {chunk['page']}]\n{chunk['text']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Шаг 3: Ответ — либо через LLM, либо только найденные фрагменты
        if self.use_llm and self.llm:
            print(f"🤖 Генерирую ответ с помощью LLM...")
            HumanMessage = self._HumanMessage
            SystemMessage = self._SystemMessage
            messages = [
                SystemMessage(content=f"""Ты помощник, который отвечает на вопросы по учебным материалам.

Твоя задача:
1. Использовать ТОЛЬКО информацию из предоставленных фрагментов конспекта
2. Если информации недостаточно, честно сказать об этом
3. Указывать номера страниц, откуда взята информация
4. Давать краткие и точные ответы

Фрагменты конспекта:
{context}"""),
                HumanMessage(content=question)
            ]
            response = self.llm.invoke(messages)
            answer = response.content
        else:
            # Режим без OpenAI: показываем только найденные фрагменты с номерами страниц
            print("📋 Формирую ответ из найденных фрагментов...")
            answer = (
                "**Вот релевантные фрагменты из конспекта (используйте их для ответа на вопрос):**\n\n"
                + context.replace("[Фрагмент", "**Фрагмент").replace("]\n", "**\n")
                + "\n\n---\n*Режим «Только поиск»: ответ не генерируется моделью. Используйте фрагменты выше.*"
            )
        
        # Шаг 4: Формируем результат
        result = {
            "answer": answer,
            "sources": [
                {
                    "page": chunk["page"],
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "score": chunk["score"]
                }
                for chunk in similar_chunks
            ],
            "question": question
        }
        
        print(f"✅ Ответ сгенерирован")
        return result
    
    def format_answer_with_sources(self, result: Dict) -> str:
        """
        Форматирует ответ с красивым отображением источников.
        
        Args:
            result: Результат от generate_answer()
            
        Returns:
            Отформатированная строка с ответом и источниками
        """
        answer_text = result["answer"]
        
        # Добавляем информацию об источниках
        sources_text = "\n\n📚 Источники:\n"
        for i, source in enumerate(result["sources"], 1):
            sources_text += f"{i}. Страница {source['page']} (релевантность: {source['score']:.2f})\n"
        
        return answer_text + sources_text


# Пример использования
if __name__ == "__main__":
    pass  # Тестирование: см. example_usage.py или app.py
