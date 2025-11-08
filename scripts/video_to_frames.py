import cv2
import numpy as np
from pathlib import Path
import sys

def extract_frames(video_path, output_dir, num_frames=15):
    """영상에서 균등하게 프레임 추출"""
    print(f"  처리 중: {video_path.name}...")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps if fps > 0 else 0
    
    if total_frames == 0:
        print("영상을 읽을 수 없음")
        cap.release()
        return 0
    
    print(f"영상: {duration:.1f}초, {total_frames}프레임")
    
    interval = max(1, total_frames // num_frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted = 0
    for i in range(num_frames):
        frame_idx = min(i * interval, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            output_path = output_dir / f"frame_{i:03d}.jpg"
            # 한글 경로 문제 해결: imencode 사용
            is_success, buffer = cv2.imencode('.jpg', frame)
            if is_success:
                with open(str(output_path), 'wb') as f:
                    f.write(buffer)
                extracted += 1
            else:
                print(f"    ✗ 저장 실패: {output_path}")
    
    cap.release()
    print(f"{extracted}개 프레임 추출 완료\n")
    return extracted

def main():
    root_dir = Path(__file__).parent.parent
    video_dir = root_dir / "data" / "videos"
    output_base = root_dir / "data" / "dataset"

    # 영상 찾기
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    
    # 처리
    total = 0
    for idx, video in enumerate(video_files, 1):
        print(f"[{idx}/{len(video_files)}] {video.stem}")
        output_dir = output_base / video.stem
        count = extract_frames(video, output_dir, num_frames=15)
        total += count
    
    print(f"\n총 {total}개 프레임 추출")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"오류: {e}")
        sys.exit(1)
