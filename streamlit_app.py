import streamlit as st
import requests
import os
import sys
from main import index_videos
import config

st.set_page_config(
    page_title="Умный поиск видео",
    page_icon="🎬",
    layout="wide"
)

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

st.title("Умный поиск видеороликов")

# API URL - используем переменную окружения или имя сервиса в docker-compose
API_URL = os.getenv("API_URL", "http://localhost:8000")
tab1, tab2 = st.tabs(["Поиск видео", "Загрузка видео"])

# Вкладка 1: Поиск видео
with tab1:

    st.header("Поиск видео")
    query = st.text_input("Введите запрос для поиска")

    if st.button("Искать", key="search_button") or query:
        if query:
            st.subheader(f"Результаты поиска по запросу: '{query}'")
            
            response = requests.post(
                f"{API_URL}/search",
                json={"query": query, "limit": config.SEARCH_LIMIT}
            )
            
            if response.status_code == 200:
                search_results = response.json()

                if config.DEBUG_MODE:
                    with st.expander("Детали ответа"):
                        st.code(response)
                        st.code(search_results)
                
                if search_results:
                    for i, result in enumerate(search_results):
                        col1, col2 = st.columns([1, 2])
                        
                        video_path = result.get('video_path', '')
                        video_name = result.get('video_name', f'Видео #{i+1}')
                        preview_path = result.get('preview_path', '')
                        transcript = result.get('transcript', '')
                        score = result.get('score', 0)
                        
                        with col1:
                            if video_path:
                                # Если путь начинается с /app, исправляем его для локальной системы
                                if video_path.startswith('/app/'):
                                    video_file_path = video_path.replace('/app/', '')
                                else:
                                    video_file_path = video_path
                                
                                try:
                                    if os.path.exists(video_file_path):
                                        with open(video_file_path, 'rb') as video_file:
                                            video_bytes = video_file.read()
                                        st.video(video_bytes)
                                    else:
                                        st.warning(f"Видео не найдено по пути: {video_file_path}")
                                except Exception as e:
                                    st.error(f"Ошибка при воспроизведении видео: {str(e)}")
                                    st.write(f"Пути к видео: оригинальный={video_path}, локальный={video_file_path}")
                            else:
                                st.warning("Видео не найдено")
                        
                        with col2:
                            if config.DEBUG_MODE:
                                st.markdown(f"**{video_name}** (Ранг: {score:.2f})")
                            st.markdown(f"**{video_name}**")
                            if transcript:
                                max_length = 150
                                display_transcript = transcript[:max_length] + "..." if len(transcript) > max_length else transcript
                                if config.DEBUG_MODE:
                                    st.write(f"Транскрипт: {display_transcript}")
                        
                        st.markdown("---")
            else:
                st.error(f"Ошибка при поиске: {response.status_code}")
        else:
            st.info("Введите запрос в поле поиска и нажмите 'Искать'.")

# Вкладка 2: Загрузка видео
with tab2:
    st.header("Загрузка видео")
    st.write("Загрузите ваши видеофайлы для последующего поиска по ним.")
    
    with st.form("upload_form", clear_on_submit=True):
        # Используем абсолютные пути для Docker
        upload_dir = os.path.join("/app", "video_examples_raw")
        videos_dir = os.path.join("/app", "video_examples")
        
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        new_uploads = st.file_uploader("Выберите видеофайлы", 
                                        type=["mp4", "avi", "mov", "mkv", "webm"], 
                                        accept_multiple_files=True,
                                        key=f"file_uploader_{st.session_state.file_uploader_key}")
        submit_button = st.form_submit_button("Загрузить файлы")
        
        if submit_button and new_uploads:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            st.session_state.uploaded_files = new_uploads
            for i, uploaded_file in enumerate(new_uploads):
                progress = (i + 1) / len(new_uploads)
                progress_bar.progress(progress)
                status_text.text(f"Загрузка {i+1} из {len(new_uploads)}: {uploaded_file.name}")
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            status_text.text(f"Загружено {len(new_uploads)} файлов")
            if config.DEBUG_MODE:
                st.success(f"Видеофайлы успешно загружены в директорию {upload_dir}")
            
            st.subheader("Загруженные файлы:")
            for file in os.listdir(upload_dir):
                st.write(f"- {file}")
        
        
    # Опция запуска индексации
    st.subheader("Индексация видео")
       
    col1, col2 = st.columns(2)
    with col1:
        if config.DEBUG_MODE:
            force_reindex = st.checkbox("Принудительная переиндексация",
                                        value=False,
                                        help="Переиндексировать все видео, даже если они уже проиндексированы")
        else:
            force_reindex = False

    with col2:
        delete_source = True
        
    if st.button("Запустить индексацию"):
        with st.spinner("Индексация видео..."):
            # Перенаправляем вывод, чтобы показать логи в Streamlit
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    index_videos(videos_dir, force_reindex)
                    success = True
                except Exception as e:
                    success = False
                    error_message = str(e)
            output = f.getvalue()
                    
            if success:
                st.success("Индексация успешно завершена!")
                        
                if delete_source and os.path.exists(upload_dir):
                    deleted_count = 0
                    failed_count = 0
                            
                    for file in os.listdir(upload_dir):
                        source_path = os.path.join(upload_dir, file)
                        try:
                            os.remove(source_path)
                            deleted_count += 1
                        except Exception as e:
                            st.error(f"Не удалось удалить файл {file}: {str(e)}")
                            failed_count += 1
                            
                    if deleted_count > 0:
                        st.success(f"Удалено {deleted_count} исходных файлов")
                    if failed_count > 0:
                        st.warning(f"Не удалось удалить {failed_count} файлов")
                        
                if config.DEBUG_MODE:
                    with st.expander("Подробности индексации"):
                        st.code(output)
            else:
                st.error(f"Ошибка при индексации видео: {error_message}")
                if config.DEBUG_MODE:
                    with st.expander("Детали ошибки"):
                        st.code(output)
                        st.code(error_message)

# Информация в сайдбаре
with st.sidebar:
    st.title("О приложении")
    st.write("Это приложение для умного поиска видеороликов по текстовому запросу.")
    
    st.header("Инструкция")
    st.write('''Поиск видео:
1. Введите текстовый запрос в поле поиска и нажмите ENTER/конпку "Искать"
2. Просмотрите найденные видеоролики
3. Воспроизведите видео прямо в браузере''')
    st.write('''Загрузка видео:
1. Нажмите на кнопку "Browse files" и выберите видео для загрузки
2. Для загрузки видео нажмите кнопку "Загрузить файлы"
3. После успешной загрузки, нажмите на кнопку "Запустить индексацию" и дождитесь сообщения "Индексация успешно завершена!"''')