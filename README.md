# StylingByte

## 📋 프로젝트 개요
**개발 기간**: 2024.01 ~ 2024.07 (6개월)  
**개발 인원**: 3명 팀 프로젝트 (개인 담당: 2차 추천모델 & Unity 연동)  
**목표**: 다단계 AI 모델을 활용한 개인화 패션 추천 시스템 구현

## 🎯 개인 담당 부분 및 성과
- ✅ **NCF 모델 구현**: Neural Collaborative Filtering 기반 2차 추천 시스템
- ✅ **Python-Unity 연동**: 크로스 플랫폼 데이터 통신 구현
- ✅ **JSON 데이터 파이프라인**: AI 모델 결과를 Unity로 전달하는 시스템 구축
- ✅ **6개월 장기 협업**: 팀 프로젝트 완주 및 모듈 통합 경험

## 🛠️ 기술 스택
- **AI/ML**: Python, TensorFlow, scikit-learn
- **추천 알고리즘**: Collaborative Filtering, NCF (Neural Collaborative Filtering)
- **Unity 연동**: C#, Process 통신, JSON 데이터 교환
- **개발환경**: Google Colab, Unity 2022.3 LTS

## 🔧 개인 구현 부분

### 1. NCF (Neural Collaborative Filtering) 모델
```python
# ai_cody_result.py - 2차 추천 모델 구현
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense

def build_ncf_model(num_users, num_items, embedding_dim=50):
    # 사용자 임베딩
    user_input = Input(shape=(), name='user_id')
    user_embed = Embedding(num_users, embedding_dim)(user_input)
    user_vec = Flatten()(user_embed)
    
    # 아이템 임베딩  
    item_input = Input(shape=(), name='item_id')
    item_embed = Embedding(num_items, embedding_dim)(item_input)
    item_vec = Flatten()(item_embed)
    
    # 신경망 기반 추천 점수 계산
    concat = Concatenate()([user_vec, item_vec])
    dense1 = Dense(128, activation='relu')(concat)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(1, activation='sigmoid')(dense2)
    
    model = Model(inputs=[user_input, item_input], outputs=output)
    return model
```

### 2. Unity 연동 시스템
```csharp
// PythonRunner.cs - Python AI 모델 실행 및 결과 수신
public class PythonRunner : MonoBehaviour {
    [SerializeField] public LoadJsonData loadJsonData;
    
    void Start() {
        // Python 프로세스 실행하여 AI 추천 모델 호출
        RunPythonScript();
        
        // JSON 결과 파일 로드 및 Unity 환경에 표시
        if (loadJsonData != null) {
            loadJsonData.LoadJson();
        }
    }
    
    private void RunPythonScript() {
        ProcessStartInfo start = new ProcessStartInfo();
        start.FileName = "python";
        start.Arguments = "ai_cody_result.py";
        start.UseShellExecute = false;
        start.CreateNoWindow = true;
        
        using (Process process = Process.Start(start)) {
            process.WaitForExit();
        }
    }
}
```

### 3. JSON 데이터 파이프라인
```csharp
// LoadJsonData.cs - AI 결과 데이터 Unity 로드
[System.Serializable]
public class ClothingItem {
    public string product;
    public List<string> most_similar;
    public float prediction_score;
}

public void LoadJson() {
    string filePath = Path.Combine(Application.dataPath, 
                                   "recommended_clothes_with_data.json");
    string jsonData = File.ReadAllText(filePath);
    
    ClothingItem[] itemsArray = JsonHelper.FromJson<ClothingItem>(jsonData);
    
    foreach (ClothingItem item in itemsArray) {
        Debug.Log($"Product: {item.product}, Score: {item.prediction_score}");
    }
}
```

## 📊 시스템 아키텍처

### 3단계 추천 파이프라인 (팀 전체)
```
1. 사용자 정보 추출 (팀원 A)
   ├── CNN 기반 성별 예측
   └── CNN 기반 나이 예측

2. 1차 후보 생성 (팀원 B)  
   ├── Collaborative Filtering
   └── 계절/스타일 필터링

3. 2차 정밀 추천 (개인 담당)
   ├── NCF 모델 학습 및 예측
   ├── 상위 3개 추천 필터링
   └── JSON 결과 생성
```

### 개인 담당 데이터 플로우
```
사용자 프로필 + 1차 후보군
          ↓
    NCF 모델 예측
          ↓
   상위 3개 선별
          ↓
    JSON 파일 저장
          ↓
   Unity Process 호출
          ↓
    Unity 환경 표시
```

## 🔬 기술적 도전과 해결

### 1. Cold Start 문제 해결 시도
**도전**: 신규 사용자에 대한 추천 정확도 저하와 충분한 상호작용 데이터 부족  
**시도한 해결 방법**: 
- **하이브리드 접근법**: 사용자 이미지 기반 특성 추출과 아이템 기반 필터링 조합
- **인구통계학적 보조**: 나이, 성별 정보를 활용한 초기 선호도 추정
- **가중치 조정**: 신규 사용자에게는 인기도 기반, 기존 사용자에게는 개인화 기반 추천 비율 조정

**학습 성과**: 추천시스템의 근본적 한계와 실무에서 사용되는 다양한 해결 전략 이해

### 2. 실시간 통신 시스템 설계 도전
**도전**: Python AI 모델과 Unity 간의 효율적 데이터 통신 구조 설계  
**현재 구현의 한계**: Process 기반 배치 처리로 인한 사용자 경험 제약
```python
# 현재 구현: 파일 기반 비동기 처리
def save_recommendation_result():
    with open('recommended_clothes_with_data.json', 'w') as f:
        json.dump(recommendation_result, f)
```

**시도한 개선 방안**:
- **WebSocket 연동 검토**: 실시간 양방향 통신을 위한 아키텍처 설계
- **REST API 설계**: Flask 기반 HTTP API 서버 구조 연구
- **비동기 처리**: Unity 코루틴과 Python 비동기 처리 연동 방법 탐구

**설계한 개선 아키텍처**:
```
Unity Client ←→ Flask API Server ←→ ML Model
     ↓              ↓                 ↓
  실시간 UI      JSON REST API     TensorFlow
```

### 3. 데이터 품질 및 모델 성능 최적화
**도전**: 제한된 학습 데이터로 인한 추천 정확도 한계  
**시도한 해결책**:
- **데이터 증강**: 기존 사용자 패턴을 분석하여 가상 사용자 프로필 생성
- **하이퍼파라미터 튜닝**: NCF 모델의 임베딩 차원, 학습률, 정규화 파라미터 최적화
- **앙상블 접근**: Collaborative Filtering과 Content-based 결과를 가중 평균으로 결합

**정량적 개선 시도**:
- 임베딩 차원 50 → 128로 증가하여 표현력 향상
- Dropout 0.2 적용으로 과적합 방지
- 학습률 스케줄링으로 수렴 안정성 개선

## 🎯 실무 연계성

### AI 모델 구현 경험
- **추천시스템 이해**: Collaborative Filtering부터 Neural 방식까지 단계적 학습
- **TensorFlow 활용**: 실제 동작하는 딥러닝 모델 구현 및 학습
- **데이터 전처리**: 실제 데이터의 노이즈와 품질 문제 경험

### 시스템 통합 능력
- **크로스 플랫폼 연동**: Python AI 모델과 C# Unity 환경 간 데이터 전달
- **JSON 데이터 처리**: 구조화된 데이터 교환 프로토콜 설계
- **비동기 처리**: 무거운 AI 연산과 실시간 UI 간의 분리

## 📝 프로젝트 한계 및 개선점

### 현재 한계
1. **하드코딩된 경로**: 개발 환경에 종속적인 절대 경로 사용
```python
# 개선 필요: 하드코딩된 경로
user_image_path = 'C:/Users/disse/OneDrive/Desktop/styling/...'
```

2. **단방향 통신**: Unity → Python → Unity의 일방향적 배치 처리
3. **데이터 품질**: 학습용 가상 데이터로 인한 추천 정확도 한계

### 향후 개선 방향
1. **API 기반 통신**: Flask/FastAPI를 활용한 RESTful API 구현
2. **환경 설정 개선**: 상대 경로 및 환경변수 활용으로 이식성 향상
3. **실시간 피드백**: 사용자 평가를 모델 학습에 반영하는 시스템

## 🎓 학습 성과 및 가치

### 추천시스템 전문성
- **이론의 실제 적용**: 논문의 NCF 알고리즘을 실제 구현
- **모델 성능 이해**: 정확도, 다양성, 신규성 간의 트레이드오프 경험
- **실무 데이터 문제**: Cold Start, 데이터 희소성 등 실제 문제 상황 체험

### 시스템 개발 능력
- **모듈러 설계**: 독립적인 컴포넌트를 조합하여 전체 시스템 구성
- **데이터 파이프라인**: AI 모델 결과를 사용자 인터페이스까지 전달하는 전체 흐름 구현
- **크로스 플랫폼 통합**: 서로 다른 기술 스택 간의 연동 경험

### 팀 협업 경험
- **장기 프로젝트 관리**: 6개월간의 지속적인 개발 및 통합 과정
- **역할 분담**: 개별 전문 영역을 담당하면서도 전체 시스템 이해
- **의사소통**: 기술적 이슈 공유 및 해결책 논의 과정

## 🔄 개발 과정에서의 성장

### 문제 해결 과정
1. **초기 구현**: 기본적인 Collaborative Filtering 구현
2. **성능 개선**: NCF 도입으로 추천 정확도 향상 시도
3. **시스템 통합**: Python 모델과 Unity UI 연동
4. **실무 고민**: 실제 배포 시 고려사항 및 개선 방향 인식

### 기술적 깊이 확장
- **머신러닝 이론**: 추천시스템의 다양한 접근 방식 이해
- **소프트웨어 아키텍처**: 확장 가능한 시스템 설계 고민
- **사용자 경험**: AI 기술을 실제 사용자가 체감할 수 있는 형태로 구현

## 💼 실무 적용 가능성

### AI 개발 역량
- **모델 구현**: TensorFlow를 활용한 실제 동작하는 AI 모델 개발
- **데이터 처리**: 실제 데이터의 전처리 및 품질 관리 경험
- **성능 평가**: 추천시스템 평가 지표 이해 및 적용

### 시스템 개발 경험
- **풀스택 이해**: AI 백엔드부터 사용자 인터페이스까지 전체 파이프라인
- **통합 능력**: 서로 다른 기술 스택을 하나의 시스템으로 연결
- **실무 제약 이해**: 이론적 완성도와 실제 구현 간의 차이 인식

## 🌟 프로젝트의 의의

### 종합적 학습 경험
이 프로젝트는 **AI 모델 개발, 시스템 통합, 팀 협업**의 세 가지 핵심 역량을 동시에 기를 수 있는 종합적인 학습 경험이었습니다. 특히 **이론적 지식을 실제 동작하는 시스템으로 구현**하는 과정에서 실무에서 마주할 수 있는 다양한 제약사항과 해결 방법을 경험할 수 있었습니다.

### 실무 준비도
- **완성도보다는 과정**: 완벽한 제품을 만들기보다는 실무에서 필요한 문제 해결 능력과 학습 태도를 기르는 데 중점
- **지속적 개선**: 현재 한계를 인식하고 향후 개선 방향을 명확히 설정
- **협업 경험**: 개별 전문성과 팀 통합 능력을 동시에 기르는 균형잡힌 개발 경험

## 📁 프로젝트 구조
```
StylingByte/
├── styling/StylingByte_model/
│   ├── _data/
│   │   └── user.csv              # 사용자 정보 데이터
│   ├── model/
│   │   ├── age_model_test.h5     # 나이 예측 모델 (팀원 구현)
│   │   ├── gender_model_test.h5  # 성별 예측 모델 (팀원 구현)
│   │   └── ncf_model.h5          # NCF 추천 모델 (개인 구현)
│   ├── src_python/
│   │   ├── ai_cody_result.py     # 개인 담당: NCF 모델 & 통합 시스템
│   │   ├── utkface_data_age_gender_predict.py  # 팀원 A 담당
│   │   └── facerecognition_for_age_gender.py   # 팀원 A 담당
│   └── src_unity/
│       ├── PythonRunner.cs       # 개인 담당: Python 프로세스 실행
│       └── LoadJsonData.cs       # 개인 담당: JSON 데이터 로드
└── README.md
```

## 🔗 참고 자료
- **GitHub Repository**: https://github.com/FrogRim/StylingByte
- **기술 문서**: NCF 구현 과정 및 Unity 연동 방법 상세 기록
- **참고 논문**: Neural Collaborative Filtering (WWW 2017)

## 💡 향후 발전 계획
1. **기술적 개선**: API 기반 실시간 통신으로 시스템 아키텍처 개선
2. **성능 최적화**: 더 정교한 하이퍼파라미터 튜닝 및 모델 개선
3. **사용자 경험**: 실제 사용자 피드백을 반영한 지속적 학습 시스템 구축
4. **확장성**: 다양한 추천 알고리즘을 쉽게 추가할 수 있는 플러그인 구조 설계
``` 
