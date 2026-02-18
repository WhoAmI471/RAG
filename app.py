"""
Главный файл Streamlit приложения для RAG чат-бота

Это простой веб-интерфейс, который позволяет:
1. Загружать PDF файлы
2. Задавать вопросы
3. Получать ответы с указанием страниц
"""

import streamlit as st
import os
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_chain import RAGChain
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настройка страницы
st.set_page_config(
    page_title="RAG Чат-бот по конспектам",
    page_icon="📚",
    layout="wide"
)

# Заголовок
st.title("📚 Чат-бот по конспектам курса")
st.markdown("""
Загрузите PDF с конспектами и задавайте вопросы! Бот найдет нужную информацию и укажет страницы.
""")

# Инициализация session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Боковая панель для загрузки документов
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор модели embeddings
    use_openai = st.radio(
        "Модель для embeddings:",
        ["OpenAI (платно, качественно)", "HuggingFace (бесплатно, локально)"],
        index=1
    )
    use_openai_embeddings = use_openai == "OpenAI (платно, качественно)"
    
    # Выбор модели LLM
    llm_choice = st.selectbox(
        "Модель для ответов:",
        [
            "Только поиск (без API)",
            "DeepSeek (бесплатно)",
            "Qwen (бесплатно)",
            "OpenAI gpt-3.5-turbo",
            "OpenAI gpt-4 (требует доступ)",
        ],
        index=0,
        help="DeepSeek и Qwen — бесплатные лимиты, работают во многих регионах. gpt-4 может быть недоступен в вашем аккаунте."
    )
    use_llm = llm_choice not in ("Только поиск (без API)", "")
    # Маппинг: выбор → (provider, model_name для OpenAI)
    if llm_choice == "DeepSeek (бесплатно)":
        llm_provider, llm_model = "deepseek", "deepseek-chat"
    elif llm_choice == "Qwen (бесплатно)":
        llm_provider, llm_model = "qwen", "qwen-turbo"
    elif llm_choice == "OpenAI gpt-4 (требует доступ)":
        llm_provider, llm_model = "openai", "gpt-4"
    else:
        llm_provider, llm_model = "openai", "gpt-3.5-turbo"
    
    st.divider()
    
    st.header("📄 Загрузка документов")
    
    # Загрузка PDF файла
    uploaded_file = st.file_uploader(
        "Выберите PDF файл с конспектами",
        type=["pdf"],
        help="Загрузите PDF файл с лекциями или конспектами"
    )
    
    if uploaded_file is not None:
        # Сохраняем файл временно
        if not os.path.exists("temp"):
            os.makedirs("temp")
        
        temp_path = f"temp/{uploaded_file.name}"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🔄 Обработать и загрузить в базу", type="primary"):
            with st.spinner("Обрабатываю PDF файл..."):
                try:
                    # Шаг 1: Обработка PDF
                    processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
                    pages = processor.extract_text_from_pdf(temp_path)
                    
                    if not pages:
                        st.error("Не удалось извлечь текст из PDF. Проверьте файл.")
                    else:
                        chunks = processor.split_into_chunks(pages)
                        
                        # Шаг 2: Создание векторного хранилища
                        if st.session_state.vector_store is None:
                            # Проверяем наличие API ключа для OpenAI
                            if use_openai_embeddings:
                                api_key = os.getenv("OPENAI_API_KEY")
                                if not api_key:
                                    st.error("⚠️ Необходим OPENAI_API_KEY в файле .env")
                                    st.stop()
                            
                            st.session_state.vector_store = VectorStore(
                                collection_name="lecture_notes",
                                use_openai=use_openai_embeddings
                            )
                        
                        # Шаг 3: Добавление документов в базу
                        st.session_state.vector_store.clear_collection()  # Очищаем старые данные
                        st.session_state.vector_store.add_documents(chunks)
                        
                        # Шаг 4: Создание RAG цепочки
                        from rag_chain import LLM_PROVIDERS
                        api_key = None
                        if use_llm:
                            env_var = LLM_PROVIDERS[llm_provider][0]
                            api_key = os.getenv(env_var)
                            if not api_key:
                                st.error(f"⚠️ Для «{llm_choice}» нужен {env_var} в файле .env. Или выберите «Только поиск».")
                                st.stop()
                        
                        try:
                            st.session_state.rag_chain = RAGChain(
                                vector_store=st.session_state.vector_store,
                                model_name=llm_model,
                                use_llm=use_llm,
                                openai_api_key=api_key,
                                llm_provider=llm_provider,
                            )
                        except Exception as e:
                            error_msg = str(e)
                            if "model" in error_msg.lower() and ("not found" in error_msg.lower() or "404" in error_msg or "does not exist" in error_msg.lower()):
                                st.error(
                                    f"❌ Модель «{llm_model}» недоступна в вашем аккаунте или регионе.\n\n"
                                    f"**Решение:**\n"
                                    f"- Выберите другую модель (например, «gpt-3.5-turbo» или «DeepSeek (бесплатно)»)\n"
                                    f"- Или используйте режим «Только поиск (без API)»"
                                )
                            elif "api" in error_msg.lower() and "key" in error_msg.lower():
                                st.error(f"❌ {error_msg}")
                            else:
                                st.error(f"❌ Ошибка при инициализации модели: {error_msg}")
                            st.stop()
                        
                        st.session_state.documents_loaded = True
                        st.success(f"✅ Загружено {len(chunks)} фрагментов из {len(pages)} страниц!")
                        
                        # Удаляем временный файл
                        os.remove(temp_path)
                
                except Exception as e:
                    st.error(f"Ошибка при обработке: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    # Информация о текущем состоянии
    st.divider()
    st.header("ℹ️ Статус")
    if st.session_state.documents_loaded:
        st.success("✅ Документы загружены")
        if st.session_state.vector_store:
            count = st.session_state.vector_store.collection.count()
            st.info(f"📊 В базе: {count} фрагментов")
    else:
        st.warning("⚠️ Документы не загружены")

# Основная область для чата
if st.session_state.documents_loaded and st.session_state.rag_chain:
    st.header("💬 Задайте вопрос")
    
    # История чата
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Показываем историю сообщений
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Показываем источники для ответов бота
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Источники"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}. Страница {source['page']}** (релевантность: {source['score']:.2f})")
                        st.text(source['text'])
    
    # Поле ввода вопроса
    if prompt := st.chat_input("Введите ваш вопрос..."):
        # Добавляем вопрос пользователя в историю
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Генерируем ответ
        with st.chat_message("assistant"):
            with st.spinner("Ищу информацию и генерирую ответ..."):
                try:
                    result = st.session_state.rag_chain.generate_answer(prompt, n_chunks=3)
                    
                    # Показываем ответ
                    st.markdown(result["answer"])
                    
                    # Показываем источники
                    with st.expander("📚 Источники"):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"**{i}. Страница {source['page']}** (релевантность: {source['score']:.2f})")
                            st.text(source['text'])
                    
                    # Сохраняем в историю
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
                except Exception as e:
                    error_msg = str(e)
                    if "model" in error_msg.lower() and ("not found" in error_msg.lower() or "404" in error_msg or "does not exist" in error_msg.lower()):
                        st.error(
                            f"❌ Модель недоступна. Выберите другую модель в настройках или используйте «Только поиск (без API)»."
                        )
                    else:
                        st.error(f"❌ Ошибка при генерации ответа: {error_msg}")
                    # Не добавляем в историю при ошибке
    
    # Кнопка очистки истории
    if st.button("🗑️ Очистить историю"):
        st.session_state.messages = []
        st.rerun()

else:
    # Инструкции, если документы не загружены
    st.info("""
    👋 Добро пожаловать!
    
    Чтобы начать работу:
    1. В боковой панели выберите модель для embeddings
    2. Загрузите PDF файл с конспектами
    3. Нажмите "Обработать и загрузить в базу"
    4. После загрузки задавайте вопросы!
    
    **Важно:** Для работы с OpenAI моделями создайте файл `.env` с вашим API ключом:
    ```
    OPENAI_API_KEY=your_api_key_here
    ```
    """)
    
    # Примеры вопросов
    st.subheader("💡 Примеры вопросов, которые можно задать:")
    example_questions = [
        "Что такое машинное обучение?",
        "Объясни концепцию нейронных сетей",
        "Какие есть типы алгоритмов обучения?",
        "Что написано на странице 5?",
    ]
    
    for q in example_questions:
        st.markdown(f"- {q}")
