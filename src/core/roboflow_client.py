"""
Roboflow 모델 클라이언트
"""
import sys
import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

# UTF-8 설정 (Windows)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from roboflow import Roboflow

class SignLanguageDetector:
    """수어 감지 모델"""
    
    # 클래스 ID → 한글 매핑
    CLASS_MAPPING = {
        0: "난자",
        1: "배란",
        2: "배아",
        3: "부인과 전문의",
        4: "여성",
        5: "인공호흡기",
        6: "중독자"
    }
    
    def __init__(self, api_key: str = None, workspace: str = None, 
                 project: str = None, version: int = None):

        # .env에서 설정 불러오기
        if not api_key:
            load_dotenv()
            api_key = os.getenv("ROBOFLOW_API_KEY")
            workspace = os.getenv("ROBOFLOW_WORKSPACE")
            project = os.getenv("ROBOFLOW_PROJECT", "medical-sign-language-2")
            version = int(os.getenv("ROBOFLOW_VERSION", "2"))

        # 모델 로드
        rf = Roboflow(api_key=api_key)
        
        if workspace:
            proj = rf.workspace(workspace).project(project)
        else:
            proj = rf.workspace().project(project)
        
        self.model = proj.version(version).model
    
    def predict_single(self, image_path: str, confidence: int = 40) -> Dict:
        try:
            # Object Detection 모델 시도
            result = self.model.predict(image_path, confidence=confidence).json()
        except TypeError:
            # Classification 모델인 경우
            print("Classification 모델 -> Object Detection으로 변경 예정")
            result = self.model.predict(image_path).json()
            
            # Classification 결과를 Object Detection 형식으로 변환
            if "predictions" in result and isinstance(result["predictions"], dict):
                # Classification 형식: {"predictions": {"class1": 0.9, "class2": 0.1}}
                pred_dict = result["predictions"]
                
                # confidence 필터링
                filtered = [
                    {
                        "class": cls,
                        "confidence": conf,
                        "class_id": self._get_class_id(cls)
                    }
                    for cls, conf in pred_dict.items()
                    if conf >= (confidence / 100)
                ]
                
                # 신뢰도 높은 순으로 정렬
                filtered.sort(key=lambda x: x["confidence"], reverse=True)
                
                result = {"predictions": filtered}
        
        # class_id를 한글로 변환
        if "predictions" in result:
            for pred in result["predictions"]:
                class_id = pred.get("class_id")
                pred["korean_name"] = self.CLASS_MAPPING.get(
                    class_id, 
                    pred.get("class", "알 수 없음")
                )
        
        return result
    
    def _get_class_id(self, class_name: str) -> int:
        """클래스 이름으로 ID 찾기"""
        # 한글 이름으로 찾기
        for class_id, korean_name in self.CLASS_MAPPING.items():
            if korean_name == class_name:
                return class_id
        # 찾지 못하면 -1
        return -1
    
    def predict_frames(self, frame_paths: List[str], 
                      confidence: int = 40) -> List[Dict]:
        results = []
        
        for frame_path in frame_paths:
            try:
                result = self.predict_single(frame_path, confidence)
                results.append(result)
            except Exception as e:
                print(f"프레임 예측 실패: {e}")
                results.append({"predictions": []})
        
        return results
    
    def aggregate_predictions(self, results: List[Dict], 
                            min_confidence: float = 0.5) -> Dict:

        # 감지된 수어 카운트
        sign_counts = {}
        sign_confidences = {}
        
        for result in results:
            for pred in result.get("predictions", []):
                korean_name = pred.get("korean_name")
                confidence = pred.get("confidence", 0)
                
                if korean_name and confidence >= min_confidence:
                    sign_counts[korean_name] = sign_counts.get(korean_name, 0) + 1
                    
                    if korean_name not in sign_confidences:
                        sign_confidences[korean_name] = []
                    sign_confidences[korean_name].append(confidence)
        
        # 평균 신뢰도 계산
        aggregated = []
        for sign, count in sign_counts.items():
            avg_confidence = sum(sign_confidences[sign]) / len(sign_confidences[sign])
            aggregated.append({
                "sign": sign,
                "count": count,
                "avg_confidence": avg_confidence,
                "frequency": count / len(results)  # 전체 프레임 중 출현 비율
            })
        
        # 빈도순 정렬
        aggregated.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "total_frames": len(results),
            "detected_signs": [item["sign"] for item in aggregated],
            "details": aggregated
        }
