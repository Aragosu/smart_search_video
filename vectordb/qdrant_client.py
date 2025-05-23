import os
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVector,
    PointStruct,
    SparseVectorParams,
    Prefetch,
    FusionQuery,
    Fusion
)
import config
import time

class QdrantManager:
    '''класс для работы с qdrant'''
    
    def __init__(self, max_retries=3, retry_delay=2):
        '''
        инициализация клиента qdrant и создание коллекции если не существует
        параметры:
            max_retries: макс кол-во попыток подключения
            retry_delay: задержка между попытками подключения (сек.)
        '''
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.collection_name = config.QDRANT_COLLECTION
        
        for attempt in range(1, max_retries + 1):
            try:
                if attempt == max_retries:
                    print("Используем встроенную версию Qdrant в памяти")
                    self.client = QdrantClient(":memory:")
                else:
                    self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
                
                # инициализируем коллекцию
                self._initialize_collections()
                print(f"Подключение к Qdrant успешно установлено с {attempt}/{max_retries} попыток")
                break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    print(f"Не удалось подключиться к Qdrant после {max_retries} попыток, используем встроенную версию")
                    self.client = QdrantClient(":memory:")
                    self._initialize_collections()
    
    def _initialize_collections(self):
        '''инициализация коллекции с тремя векторными пространствами'''
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "visual": VectorParams(
                            size=config.VISUAL_VECTOR_SIZE,
                            distance=Distance.COSINE
                        ),
                        "text_dense": VectorParams(
                            size=config.TEXT_VECTOR_SIZE,
                            distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "text_sparse": SparseVectorParams(),
                    }
                )
            
            if 'semantic_cache_queries' not in collection_names:
                self.client.create_collection(
                    collection_name='semantic_cache_queries',
                    vectors_config={
                        "text_dense_vector": VectorParams(
                            size=1024,
                            distance="Cosine")
                    }
                )

                print(f"Коллекция '{self.collection_name}' успешно создана")
            else:
                print(f"Используется существующая коллекция '{self.collection_name}'")
        except Exception as e:
            print(f"Ошибка при инициализации коллекции: {str(e)}")
            raise
    
    def index_video(self, video_path: str,
                    visual_embeds: np.ndarray,
                    text_dense_embeds: np.ndarray,
                    text_sparse_embeds,
                    metadata: Dict[str, Any]) -> str:
        '''
        индексирование видео в qdrant (3 типа эмбеддингов)
        параметры:
            video_path: путь к видеофайлу
            visual_embeds: визуальные эмбеддинги для видео
            text_dense_embeds: текстовые эмбеддинги (из транскрипции)
            text_sparse_embeds:
            metadata: метаданные видео
        вывод: ID созданной точки в коллекции
        '''
        try:
            point_id = str(uuid.uuid4())
            video_name = os.path.basename(video_path)
            
            if not video_path.startswith('/app/'):
                if os.path.isabs(video_path):
                    normalized_path = os.path.join('/app', os.path.basename(video_path))
                else:
                    normalized_path = os.path.join('/app', video_path)
            else:
                normalized_path = video_path
            
            metadata.update({
                "video_path": normalized_path,
                "video_name": video_name
            })
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={
                            "visual": visual_embeds.tolist(),
                            "text_dense": text_dense_embeds.tolist(),
                            "text_sparse": SparseVector(
                                indices=text_sparse_embeds.indices,
                                values=text_sparse_embeds.values,
                            )
                        },
                        payload=metadata
                    )
                ]
            )
            
            return point_id
        except Exception as e:
            print(f"Ошибка при индексации видео {video_path} в Qdrant: {str(e)}")
            raise
    
    def upsert_semantic_cache(self, query_text: str, query_vector: List[float], metadata: List[Dict[str, Any]]):
        try:
            point_id = str(uuid.uuid4())

            metadata_cache = {
                "query_text": query_text,
                "metadata": metadata
            }

            self.client.upsert(
                collection_name='semantic_cache_queries',
                points=[
                    PointStruct(
                        id=point_id,
                        vector={"text_dense_vector": query_vector},
                        payload=metadata_cache
                    )
                ]
            )
        except Exception as e:
            print(f"Ошибка при семантическом кэшировании: {str(e)}")
            return []

    def semantic_search(self, query_vector: np.ndarray):
        '''
        семантический поиск видео
        параметры:
            query_vector: вектор запроса
        вывод: найденный семантический запрос
        '''
        try:
            search_result = self.client.search(
                collection_name='semantic_cache_queries',
                query_vector=('text_dense_vector', query_vector.tolist()),
                limit=1
            )
                
            return search_result
        except Exception as e:
            print(f"Ошибка при семантическом поиске в Qdrant: {str(e)}")
            return []
        
    def hybrid_search_dbsf(self, query_text: str,
                           visual_vector: Optional[np.ndarray] = None, 
                           text_dense_vector: Optional[np.ndarray] = None,
                           text_sparse_vector: Optional[np.ndarray] = None,
                           limit: int = config.SEARCH_LIMIT) -> List[Dict[str, Any]]:
        '''
        мультимодальный поиск по всем эмбеддингам
        параметры:
            query_text: текстовый запрос
            visual_vector: визуальный вектор запроса
            text_dense_vector: текстовый вектор запроса - dense
            text_sparse_vector: текстовый вектор запроса - sparse
            limit: максимальное количество результатов
        вывод: список найденных видео с объединенной оценкой
        '''
        try:
            results = []
            
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=visual_vector,
                        using="visual",
                        limit=limit*2,
                    ),
                    Prefetch(
                        query=text_dense_vector.tolist(),
                        using="text_dense",
                        limit=limit*2,
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=text_sparse_vector.indices.tolist(),
                            values=text_sparse_vector.values.tolist(),
                        ),
                        using="text_sparse",
                        limit=limit*2,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
            )
            

            for elem in search_result.dict()['points']:
                results.append({'id':elem['id'],
                            'score':elem['score'],
                            'video_name':elem['payload']['video_name'],
                            'video_path':elem['payload']['video_path'],
                            'transcript':elem['payload']['transcript'],
                            'query':query_text,
                            'preview_path':elem['payload']['preview_path']})
                
            return results
        
        except Exception as e:
            print(f"Ошибка при выполнении гибридного поиска: {str(e)}")
            return []