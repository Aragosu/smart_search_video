FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/video_examples /app/video_examples_raw /app/previews /app/temp

# предварительно грузим модели
RUN echo 'import torch\n\
from transformers import CLIPProcessor, CLIPModel\n\
from sentence_transformers import SentenceTransformer\n\
from fastembed.sparse import SparseTextEmbedding\n\
import config\n\
\n\
# Предварительная загрузка моделей\n\
print("Загрузка CLIP модели...")\n\
visual_model = CLIPModel.from_pretrained(config.VISUAL_MODEL)\n\
visual_processor = CLIPProcessor.from_pretrained(config.VISUAL_MODEL)\n\
\n\
print("Загрузка Text модели...")\n\
text_model = SentenceTransformer(config.TEXT_MODEL)\n\
\n\
print("Загрузка Sparse Text модели...")\n\
text_sparse_model = SparseTextEmbedding(config.TEXT_SPARSE_MODEL)\n\
\n\
print("Все модели успешно загружены!")' > preload_models.py

RUN python preload_models.py

EXPOSE 8000 8501