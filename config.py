import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).parent
VIDEO_DIR = os.path.join(BASE_DIR, "video_examples")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# Параметры обработки видео
FRAME_EXTRACTION_INTERVAL = 1  # Интервал извлечения кадров в секундах
MAX_FRAMES_PER_VIDEO = 30  # Максимальное количество кадров для анализа с одного видео

# Параметры моделей
TRANSCRIBE_MODEL = "base"
VISUAL_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
TEXT_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
TEXT_SPARSE_MODEL = "Qdrant/bm25"

# Параметры Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = "video_search"
VISUAL_VECTOR_SIZE = 512
AUDIO_VECTOR_SIZE = 512
TEXT_VECTOR_SIZE = 1024

# Параметры API
API_HOST = "0.0.0.0"
API_PORT = 8000
SEARCH_LIMIT = 10
CACHE_MARK = False
THRESHOLD_SEMANTIC = 0.9

# Streamlit
DEBUG_MODE = False