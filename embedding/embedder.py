import torch
import numpy as np
from typing import List, Dict, Any
from transformers import CLIPProcessor, CLIPModel
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from fastembed.sparse import SparseTextEmbedding
import config

class MultimodalEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.visual_model = CLIPModel.from_pretrained(config.VISUAL_MODEL).to(self.device)
        self.visual_processor = CLIPProcessor.from_pretrained(config.VISUAL_MODEL)
        self.text_model = SentenceTransformer(config.TEXT_MODEL).to(self.device)
        self.text_sparse_model = SparseTextEmbedding(config.TEXT_SPARSE_MODEL)#.to(self.device)

    def create_visual_embeddings(self, frames: List[np.ndarray]) -> np.ndarray:
        '''создание мультимодальных эмбеддингов из фреймов (для визуальных эмбедов)'''
        embeddings = []
        
        for frame in frames:
            inputs = self.visual_processor(images=frame, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.visual_model.get_image_features(**inputs)
                
            embedding = outputs.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
            
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(config.VISUAL_VECTOR_SIZE)
    
    
    def create_text_embeddings(self, text: str) -> np.ndarray:
        '''создание текстовых dense-эмбеддингов из транскрипции'''
        if not text:
            return np.zeros(config.TEXT_VECTOR_SIZE)
            
        with torch.no_grad():
            embedding = self.text_model.encode(text)
            
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def create_text_sparse_embeddings(self, text: str):
        '''создание текстовых sparse-эмбеддингов из транскрипции'''
        if not text:
            with torch.no_grad():
                embedding = list(self.text_sparse_model.embed(''))
        else:
            with torch.no_grad():
                embedding = list(self.text_sparse_model.embed(text))
        
        return embedding
        
    def create_clip_text_embedding(self, text: str) -> np.ndarray:
        '''создание мультимодальных эмбеддингов с CLIP (для текст запросов поиска по визуальным эмбедам)'''
        if not text:
            return np.zeros(config.VISUAL_VECTOR_SIZE)
        inputs = self.visual_processor(text=text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.visual_model.get_text_features(**inputs)
            
        embedding = outputs.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding 