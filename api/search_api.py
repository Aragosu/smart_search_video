from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel
import config
from vectordb.qdrant_client import QdrantManager
from embedding.embedder import MultimodalEmbedder

class SearchQuery(BaseModel):
    '''Модель для запроса поиска видео'''
    query: str
    limit: Optional[int] = config.SEARCH_LIMIT

class SearchResult(BaseModel):
    '''Модель для результата поиска видео'''
    id: str
    score: float
    video_name: str
    video_path: str
    transcript: Optional[str] = None
    query: str
    preview_path: Optional[str] = None

def create_app():
    '''Создание FastAPI приложения для поиска видео'''
    app = FastAPI(title="Video Search API", description="API для умного поиска видеороликов")
    
    # Настройка CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    embedder = MultimodalEmbedder()
    db_manager = QdrantManager()
    
    @app.get("/", tags=["Root"])
    async def root():
        '''начальный эндпоинт'''
        return {"message": "Умный поиск видеороликов API"}
    
    @app.post("/search", response_model=List[SearchResult], tags=["Search"])
    async def search_videos(search_query: SearchQuery):
        '''
        эндпоинт для поиска видео по текстовому запросу
        параметры:
            search_query: запрос для поиска видео
        вывод: список найденных видео с оценкой
        '''
        cache_mark = config.CACHE_MARK
        try:
            # dense-эмбеддинги
            text_dense_vector = embedder.create_text_embeddings(search_query.query)
            semantic_result = db_manager.semantic_search(text_dense_vector)

        # проверка на наличие запроса в семантической кэше
            if len(semantic_result)==0:
                cache_mark = True
            else:
                search_score = semantic_result[0].score
                search_metadata = semantic_result[0].payload['metadata']
                if search_score >= config.THRESHOLD_SEMANTIC:
                    results = search_metadata
                    cache_mark = False
                else:
                    cache_mark = True

            if cache_mark:
                # мультимодальных эмбеддинги CLIP
                clip_text_embedding = embedder.create_clip_text_embedding(search_query.query)
                # sparse-эмбеддинги
                text_sparse_vector = embedder.create_text_sparse_embeddings(search_query.query)

                results = db_manager.hybrid_search_dbsf(
                    query_text=search_query.query,
                    visual_vector=clip_text_embedding,
                    text_dense_vector=text_dense_vector,
                    text_sparse_vector=text_sparse_vector[0],
                    limit=search_query.limit
                )
                db_manager.upsert_semantic_cache(search_query.query, text_dense_vector.tolist(), results)

            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}")
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        '''Эндпоинт для проверки здоровья сервиса'''
        return {"status": "healthy"}
    
    return app 