import argparse
from video_processor.processor import VideoProcessor
from embedding.embedder import MultimodalEmbedder
from vectordb.qdrant_client import QdrantManager
from qdrant_client.models import SparseVector
from api.search_api import create_app
import os



def setup_parser():
    parser = argparse.ArgumentParser(description='Умный поиск видеороликов')
    parser.add_argument('--mode', type=str, choices=['index', 'serve'], required=True,
                        help='Режим работы: index - индексация видео, serve - запуск API')
    parser.add_argument('--videos_dir', type=str, default='./video_examples',
                        help='Директория с видеофайлами для индексации')
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                        help='Хост для запуска API')
    parser.add_argument('--port', type=int, default=8000, 
                        help='Порт для запуска API')
    parser.add_argument('--force-reindex', action='store_true', 
                        help='Принудительная переиндексация всех видео, даже если они уже проиндексированы')
    return parser

def index_videos(videos_dir, force_reindex=False):
    '''
    индексация видео из указанной директории, умеет распознавать существующие видео и пропускать их
    параметры:
        videos_dir: Директория с видеофайлами
        force_reindex: Флаг для принудительной переиндексации всех видео
    '''
    # проверка директории
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir, exist_ok=True)

    processor = VideoProcessor()
    
    # Конвертация видео из raw-директории в директорию для индексации с правильным кодеком
    raw_videos_dir = os.path.join(os.path.dirname(videos_dir), "video_examples_raw")
    try:
        if os.path.exists(raw_videos_dir):
            processor.convert_videos_to_web_compatible(source_dir=raw_videos_dir, target_dir=videos_dir)
            print(f"Конвертация видеокодека успешно завершена")
        else:
            print(f"Директория с исходными видео {raw_videos_dir} не найдена. Пропускаем этап конвертации.")
    except Exception as e:
        print(f"Ошибка при конвертации видео: {str(e)}")
        
    video_paths = processor.get_video_files(videos_dir)
    print(f"Найдено {len(video_paths)} видеофайлов")
    
    embedder = MultimodalEmbedder()
    db_manager = QdrantManager()
    
    # список уже существующих видео
    indexed_videos = set()
    if not force_reindex:
        try:
            scroll_result = db_manager.client.scroll(
                collection_name=db_manager.collection_name,
                limit=1000,
                with_vectors=False
            )
            points = scroll_result[0]
            
            for point in points:
                if 'video_path' in point.payload:
                    indexed_path = os.path.normpath(point.payload['video_path'])
                    indexed_videos.add(indexed_path)
            
            print(f"Найдено {len(indexed_videos)} уже проиндексированных видео")
        except Exception as e:
            print(f"Ошибка при получении списка проиндексированных видео: {str(e)}")
    
    # только новые видео
    new_videos = []
    skipped_videos = []
    
    for video_path in video_paths:
        normalized_path = os.path.normpath(video_path)
        if normalized_path in indexed_videos and not force_reindex:
            skipped_videos.append(video_path)
        else:
            new_videos.append(video_path)
    
    print(f"Видео для индексации: {len(new_videos)}")
    print(f"Пропущено уже проиндексированных видео: {len(skipped_videos)}")
    
    # обрабатываем новые видео
    for i, video_path in enumerate(new_videos, 1):
        try:
            print(f"[{i}/{len(new_videos)}] Обработка видео: {video_path}")
            
            frames = processor.extract_frames(video_path)            
#            preview_path = processor.save_preview_image(video_path)
            
#            if preview_path and os.path.exists(preview_path):
#                preview_rel_path = os.path.join("previews", os.path.basename(preview_path))
#            else:
#                preview_rel_path = None
            
            audio_path = processor.extract_audio(video_path)
            transcript = processor.transcribe_audio(audio_path)
            
            # эмбеды
            visual_embeds = embedder.create_visual_embeddings(frames)
            text_dense_embeds = embedder.create_text_embeddings(transcript)
            text_sparse_embeds = embedder.create_text_sparse_embeddings(transcript)

            # сохраняем в БД
            video_id = db_manager.index_video(
                video_path=video_path,
                visual_embeds=visual_embeds,
                text_dense_embeds=text_dense_embeds,
                text_sparse_embeds=text_sparse_embeds[0],
                metadata={
                    "transcript": transcript,
                    "frames_count": len(frames),
                    "preview_path": '-'
                }
            )
            
            print(f"  - Видео успешно проиндексировано с ID: {video_id}")
            
            # чистим временный файл
            processor.cleanup_temp_file(audio_path)
            
        except Exception as e:
            print(f"Ошибка при обработке видео {video_path}: {str(e)}")
            if 'audio_path' in locals():
                processor.cleanup_temp_file(audio_path)
    
    print("\nИндексация завершена!")
    print(f"Всего проиндексировано: {len(new_videos)} видео")
    print(f"Пропущено: {len(skipped_videos)} видео")


def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.mode == 'index':
        index_videos(args.videos_dir, args.force_reindex)
    elif args.mode == 'serve':
        print(f"Запуск API на http://{args.host}:{args.port}")
        app = create_app()
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main() 