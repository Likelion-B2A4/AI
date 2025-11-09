"""
영상 수어 인식 테스트
"""
import sys
from pathlib import Path
import json

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.video_processor import VideoProcessor
from src.core.roboflow_client import SignLanguageDetector


def test_video_prediction(video_path: str):
    """영상으로 수어 인식 테스트"""

    
    # 1. 영상 정보 확인
    print(f"영상: {Path(video_path).name}")
    processor = VideoProcessor(num_frames=15)
    
    try:
        info = processor.get_video_info(video_path)
        print(f"   프레임: {info['total_frames']}개")
    except Exception as e:
        print(f"영상 정보 읽기 실패: {e}")
        return
    
    # 2. 프레임 추출
    try:
        frames = processor.extract_frames(video_path)
        print(f"   {len(frames)}개 프레임 추출 완료")
    except Exception as e:
        print(f"프레임 추출 실패: {e}")
        return
    
    # 3. 모델 로드
    try:
        detector = SignLanguageDetector()
    except Exception as e:
        print(f"AI 모델 로드 실패: {e}")
        return
    
    # 4. 각 프레임 예측
    frame_paths = []
    
    try:
        # 프레임을 임시 파일로 저장
        for i, frame in enumerate(frames):
            temp_path = processor.save_frame_temp(frame)
            frame_paths.append(temp_path)
        
        # 예측 수행
        results = detector.predict_frames(frame_paths, confidence=40)
        
        # 임시 파일 삭제
        import os
        for path in frame_paths:
            try:
                os.unlink(path)
            except:
                pass
        
    except Exception as e:
        print(f"예측 실패: {e}")
        return
    
    # 5. 결과 종합
    aggregated = detector.aggregate_predictions(results, min_confidence=0.4)
    
    # 6. 결과 출력
    if aggregated["detected_signs"]:
        print(f"\n감지된 수어: {', '.join(aggregated['detected_signs'])}")
        print(f"\n상세 정보:")
        
        for detail in aggregated["details"]:
            print(f"\n   🔹 {detail['sign']}")
            print(f"      출현: {detail['count']}/{aggregated['total_frames']}프레임 ({detail['frequency']*100:.1f}%)")
    else:
        print("\n수어가 감지되지 않았습니다")



if __name__ == "__main__":
    # 환자 테스트 영상 폴더
    patient_video_dir = Path("D:/Sign-Language-AI/data/patient_videos")
    
    # patient_videos에서 영상 찾기
    videos = list(patient_video_dir.glob("*.mp4")) + \
             list(patient_video_dir.glob("*.avi"))
    
    if not videos:
        print("테스트할 영상이 없음")
        sys.exit(1)
    
    video_path = str(videos[0])
    print(f"테스트 영상: {videos[0].name}\n")
    
    test_video_prediction(video_path)
