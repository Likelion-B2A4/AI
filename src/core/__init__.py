"""
core 패키지
수어 인식 AI의 핵심 모듈
"""
from .roboflow_client import SignLanguageDetector
from .video_processor import VideoProcessor

__all__ = [
    "SignLanguageDetector",
    "VideoProcessor"
]
