import os
import re
import subprocess
import tempfile
import shutil
import cv2
import numpy as np
from typing import List, Tuple
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip
from pathlib import Path
import config
import torch
import uuid
import soundfile as sf
from PIL import Image

class VideoProcessor:
    '''класс для обработки видеофайлов'''
    
    def __init__(self):
        '''инициализация процессора видео'''
        self.frame_interval = config.FRAME_EXTRACTION_INTERVAL
        self.max_frames = config.MAX_FRAMES_PER_VIDEO
        self.audio_model = WhisperModel(config.TRANSCRIBE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    
    def get_video_files(self, directory: str) -> List[str]:
        '''получение списка всех видеофайлов в директории'''
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(directory, file))
                
        return video_files
    
    def convert_videos_to_web_compatible(self, source_dir, target_dir):
        """
        Конвертирует все видео из исходной директории в веб-совместимый формат
        с использованием кодека H.264 и сохраняет их в целевую директорию 
        с сохранением оригинальных имен файлов.
        После успешной конвертации удаляет исходные файлы.
        
        Параметры:
            source_dir: Директория с исходными видеофайлами
            target_dir: Директория для сохранения конвертированных видеофайлов
        """
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_paths = []
        
        for root, _, files in os.walk(source_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_paths.append(os.path.join(root, file))
        
        # Списки для отслеживания успешно сконвертированных и пропущенных файлов
        successfully_converted = []
        skipped_files = []
        failed_files = []
        
        # Конвертируем каждое видео
        for i, source_path in enumerate(video_paths, 1):
            try:

                base_name = os.path.basename(source_path)
                file_name_without_ext = os.path.splitext(base_name)[0]
                safe_name = re.sub(r'[^\w\-_.]', '_', file_name_without_ext)
                target_path = os.path.join(target_dir, f"{safe_name}.mp4")
                
                if os.path.exists(target_path):
                    skipped_files.append(source_path)
                    continue
                
                # Создаем временную директорию для промежуточного файла
                temp_dir = tempfile.mkdtemp()
                temp_output_path = os.path.join(temp_dir, f"temp_{safe_name}.mp4")
                
                try:
                    command = [
                        'ffmpeg', 
                        '-i', source_path,
                        '-vcodec', 'libx264',
                        '-acodec', 'aac',
                        '-strict', 'experimental',
                        '-pix_fmt', 'yuv420p',
                        '-profile:v', 'baseline',
                        '-level', '3.0',
                        '-movflags', '+faststart',
                        '-y',
                        temp_output_path
                    ]
                    
                    process = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    if process.returncode == 0:
                        shutil.copy2(temp_output_path, target_path)
                        successfully_converted.append(source_path)
                    else:
                        error_message = process.stderr.decode()
                        print(f"  - Ошибка конвертации видео: {error_message}")
                        failed_files.append(source_path)
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
            except Exception as e:
                print(f"Ошибка при конвертации видео {source_path}: {str(e)}")
                failed_files.append(source_path)
        
        if successfully_converted:
            for source_path in successfully_converted:
                os.remove(source_path)

        print("\nИтоги конвертации видео:")
        print(f"  - Успешно сконвертировано и перемещено: {len(successfully_converted)}")
        print(f"  - Пропущено (уже существуют): {len(skipped_files)}")
        print(f"  - Не удалось сконвертировать: {len(failed_files)}")
  
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        '''извлечение ключевых кадров из видео с заданным интервалом'''
        frames = []
        vidcap = cv2.VideoCapture(video_path)
        
        # тут считаем интервал фреймов
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_interval_count = int(fps * self.frame_interval)
        
        frame_count = 0
        success = True
        
        while success and len(frames) < self.max_frames:
            success, image = vidcap.read()
            
            if not success:
                break
                
            if frame_count % frame_interval_count == 0:
                # Преобразование из BGR в RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames.append(image_rgb)
                
            frame_count += 1
            
        vidcap.release()
        return frames
    
    def extract_audio(self, video_path: str) -> str:
        '''извлечение аудио из видео, сохраняем во временный WAV (в temp)'''
        
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_audio_path = os.path.join(temp_dir, f"audio_{uuid.uuid4().hex}.wav")
        
        try:
            video = VideoFileClip(video_path)
            if video.audio is None:
                dummy_audio = np.zeros(16000, dtype=np.float32)
                sf.write(temp_audio_path, dummy_audio, 16000)
            else:
                video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', verbose=False, logger=None)
                if not os.path.exists(temp_audio_path):
                    raise Exception(f"Файл {temp_audio_path} не был создан")
            
            video.close()
            
            return temp_audio_path
            
        except Exception as e:
            print(f"Ошибка при извлечении аудио из {video_path}: {str(e)}")
            dummy_audio = np.zeros(16000, dtype=np.float32)
            sf.write(temp_audio_path, dummy_audio, 16000)
            print(f"Создан пустой аудиофайл {temp_audio_path}")
            return temp_audio_path
    
    def transcribe_audio(self, audio_path: str) -> str:
        '''транскрипция аудио в текст с использованием faster-whisper'''
        try:
            if not os.path.exists(audio_path):
                print(f"Предупреждение: аудиофайл {audio_path} не существует")
                return ""
            
            segments, info = self.audio_model.transcribe(audio_path, beam_size=5)
            transcript = " ".join([segment.text for segment in segments])
                
            return transcript
        
        except Exception as e:
            print(f"Ошибка при транскрипции аудио {audio_path}: {str(e)}")
            return ""
            
    def cleanup_temp_file(self, file_path: str) -> None:
        '''в проыессе создаем временный файл, функция для очистки'''
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Ошибка при удалении временного файла {file_path}: {str(e)}")
    
    def save_preview_image(self, video_path: str) -> str:
        '''
        (отключена, но вдруг пригодится в будущих версиях) создаем превьюхи, чтобы видеть на фронте
        параметры:
            video_path: путь к видео
        вывод: путь к сохраненной превьюхе
        '''
        try:
            # проверка директории
            static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
            previews_dir = os.path.join(static_dir, "previews")
            
            if not os.path.exists(static_dir):
                os.makedirs(static_dir, exist_ok=True)
                
            if not os.path.exists(previews_dir):
                os.makedirs(previews_dir, exist_ok=True)
            
            # имя превьюхи
            video_basename = os.path.basename(video_path)
            video_name_without_ext = os.path.splitext(video_basename)[0]
            preview_path = os.path.join(previews_dir, f"{video_name_without_ext}.jpg")

            if os.path.exists(preview_path):
                return preview_path

            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            
            if success:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                pil_image.save(preview_path)

                preview_status = True
                        
            else:
                # Если не удалось прочитать кадр, создаем пустое изображение
                print(f"Не удалось получить кадр из видео {video_path}, создаем пустое изображение")
                empty_image = np.zeros((360, 640, 3), dtype=np.uint8)
                pil_image = Image.fromarray(empty_image)
                pil_image.save(preview_path)
                
                preview_status = False

                if preview_status:
                    if os.path.exists(preview_path):
                        pass
                    else:
                        print(f"ОШИБКА: Файл не был создан: {preview_path}")
                else:
                    if os.path.exists(preview_path):
                        pass
                    else:
                        print(f"ОШИБКА: Пустое превью не было создано: {preview_path}")
            
            vidcap.release()
            return preview_path
            
        except Exception as e:
            print(f"ОШИБКА при создании превью для {video_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        