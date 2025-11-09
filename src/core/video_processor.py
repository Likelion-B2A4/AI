"""
영상 처리 모듈
"""
import cv2
import tempfile
from pathlib import Path
from typing import List
import numpy as np


class VideoProcessor:
    """영상을 프레임으로 추출하고 처리"""
    
    def __init__(self, num_frames: int = 15):

        self.num_frames = num_frames
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없음: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError("영상에 프레임이 없음")
        
        # 균등 간격으로 프레임 추출
        interval = max(1, total_frames // self.num_frames)
        frames = []
        
        for i in range(self.num_frames):
            frame_idx = min(i * interval, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def save_frame_temp(self, frame: np.ndarray) -> str:

        # 한글 경로 문제 해결: imencode 사용
        is_success, buffer = cv2.imencode('.jpg', frame)
        
        if not is_success:
            raise ValueError("프레임 인코딩 실패")
        
        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix='.jpg'
        )
        temp_file.write(buffer.tobytes())
        temp_file.close()
        
        return temp_file.name
    
    def get_video_info(self, video_path: str) -> dict:

        cap = cv2.VideoCapture(video_path)
        
        info = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        
        info["duration"] = (
            info["total_frames"] / info["fps"] 
            if info["fps"] > 0 else 0
        )
        
        cap.release()
        return info
