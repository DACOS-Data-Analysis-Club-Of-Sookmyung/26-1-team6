# 학생 성적 영향 요인 분석 및 맞춤형 학습 피드백 모델
초보 6팀 토이 프로젝트

---

## 프로젝트 소개
단순 성적 확인과 공부시간으로 부족한 요인을 파악하기 어렵다는 문제를 해결하기 위해서, 학생 데이터를 활용하여 성적에 영향을 미치는 주요 요인을 분석하고 머신러닝 기반 성적 예측 모델을 구축합니다.
예측 결과와 주요 영향 변수를 바탕으로 학생별 학습 상태를 파악하고 맞춤형 개선 방향을 제시하는 것을 목표로 합니다.

---

## 문제 정의
- 학생 성적은 공부 시간뿐만 아니라 출석, 학습 환경, 활동 참여 등 다양한 요소의 영향을 받음
- 단순 성적 확인만으로는 다른 학생 대비 자신의 학습 수준과 부족한 요인을 파악하기 어려움
- 데이터 기반 분석을 통해 성적 영향 요인을 찾고 개인별 개선 방향을 제시할 필요가 있음

---

## 데이터셋 설명
- kaggle Students Performance Dataset

2,392명 학생 성적 데이터 활용
(학생 기본 정보, 공부 시간, 결석 횟수, 학습 지원 여부, 비교과 활동 여부, GPA 및 성적 등급 데이터 포함)

---

## 전체 코드 구성 및 사용 기술
- 프로젝트(코드) 구성
```text
[ 학생 성적 예측 파이프라인 ]
│
├─ [1] 데이터 수집
│   ├─ kaggle Students Performance Dataset 다운로드
│   └─ csv 파일을 DataFrame으로 로드
│
├─ [2] 데이터 확인 & 정제
│   ├─ df.info() / df.describe() / df.isnull().sum()
│   └─ 불필요 컬럼(StudentID) 제거
│
├─ [3] 데이터 분석 (EDA)
│   ├─ df.hist(), sns.scatterplot(), sns.boxplot()  →  가설 3개 검증
│   ├─ sns.heatmap(df.corr())  →  Absences 상관계수 0.73 (가장 높음)
│   └─ GradeClass 클래스 불균형 확인
│
├─ [4] 데이터 전처리
│   ├─ 입력 변수(X)와 타겟 변수(y = GradeClass) 분리
│   ├─ Train/Test 데이터 분리 (8:2)
│   └─ StandardScaler로 스케일링
│
├─ [5] 모델링
│   ├─ 5-1. LogisticRegression
│   │       기본 모델, 실험 : class_weight='balanced' / max_iter 조정 / GridSearchCV
│   │
│   ├─ 5-2. RandomForestClassifier
│   │       기본 모델, 실험 : 하이퍼파라미터 튜닝 / 중요 변수만 사용한 모델 비교
│   │
│   └─ 5-3. K-Means 군집화
│           PCA(n_components=2) → 엘보우 방법 / 실루엣 점수로 k=6 결정
│           → 클러스터별 최빈 등급 매핑
│
├─ [6] 모델 성능 비교
│   └─ Accuracy / Macro F1 기준, 7개 모델 변형 비교
│       → 랜덤 포레스트 기본 모델이 최고 성능
│
└─ [7] 등급 예측 기능
    ├─ 상관관계가 가장 높은 핵심 변수 4개 : ['StudyTimeWeekly', 'Absences', 'ParentalSupport', 'Tutoring']
    ├─ predict_my_grade()  →  등급 및 확률 예측
    └─ compare_with_A_group()  →  A등급 평균 대비 리포트 생성
```

- 데이터 분석 및 전처리

  -결측치 확인 및 불필요한 변수 제거(StudentID 제거)
  
  -변수별 성적 분포 분석 및 상관관계 시각화(EDA)
  
  -데이터 스케일링 및 Train/Test 데이터 분리
  

- 사용 기술

| 종류 | 기술 |
| --- | --- |
| Python 기반 데이터 분석 | Pandas, Numpy, Matplotlib, Seaborn |
| 머신러닝 모델 | Logistic Regression, Random Forest |
| 학생 유형 분석 | K-Means Clustering |
| 핵심 요인 도출 | Feature Importance |

---

## 실행 방법
*이 프로젝트는 Google Colab 환경에서 실행하는 것을 기준으로 합니다.*

1. `Students_Team6.ipynb` 파일을 Google Colab에서 엽니다.
   - GitHub 저장소 페이지에서 파일을 열고 상단의 "Open in Colab" 배지를 클릭하거나,
   - Colab에서 '파일 → 노트북 열기 → GitHub' 탭에 저장소 주소를 입력해 엽니다.
  
2. 상단 메뉴에서 **런타임 → 모두 실행**을 클릭해 전체 셀을 순서대로 실행합니다.
   - 첫 셀에서 `kagglehub.dataset_download(...)`로 데이터셋을 바로 내려받습니다.
     * 이 데이터셋은 공개 데이터셋이라 별도의 Kaggle API 인증 없이 다운로드됩니다.
   - 만약 kagglehub 관련 오류가 발생하면 `!pip install kagglehub -q`를 노트북 맨 위에 추가해 설치합니다.

---

## 모델 성능
### [Logistic Regression]
| 실험 | 지표 | 성능 |
| --- | --- | --- |
| 기본 | Accuracy | 0.6827 |
|  | Macro F1 Score | 0.4533 |
| 클래스 불균형 처리 | Accuracy | 0.5741 |
|  | Macro F1 Score | 0.4196 |
| 수렴 안정화 | Accuracy | 0.6827 |
|  | Macro F1 Score | 0.4533 |
| C값 탐색 | Accuracy | 0.6806 |
|  | Macro F1 Score | 0.4500 |

### [Random Forest]
| 실험 | 지표 | 성능 |
| --- | --- | --- |
| **기본** | **Accuracy** | **0.6952** |
|  | **Macro F1 Score** | **0.5073** |
| 하이퍼파라미터 튜닝 | Accuracy | 0.6868 |
|  | Macro F1 Score | 0.4970 |
| 중요 변수만 사용 | Accuracy | 0.6618 |
|  | Macro F1 Score | 0.4987 |

*최종 모델은 Random Forest 기본 모델로 선정*

*예측 모델을 만들 때 모든 특성을 다 수치화하기 어려워 4개의 핵심 변수만 선정해서 최종 예측 기능 구현*

| 최종 모델 | 지표 | 성능 |
| --- | --- | --- |
| 4개 핵심 변수 사용 | Accuracy | 0.6681 |
|  | Macro F1 Score | 0.5326 |

---

## 등급 예측 기능
- 입력값
: 상관관계 Heatmap에서 상관관계가 가장 높았던 변수 4개 사용

| 입력값 | 사용 변수 |
| --- | --- |
| 주간 공부시간(0-20) | StudyTimeWeekly |
| 결석 횟수(0-30) | Absences |
| 부모지원 수준(0:없음 ~ 4:매우 높음) | ParentalSupport |
| 튜터링 여부(0:아니오, 1:예) | Tutoring |

- 등급 예측 및 등급별 확률 표시

- A등급(평균)과 내 입력값과 차이 비교

---

## 대시보드(추가 예정)

(EDA·모델 성능·핵심 변수를 한눈에 볼 수 있는 대시보드를 추가할 예정입니다.)

---
  
## 결과 및 기대 효과
- 학생의 예상 성적 수준 및 다른 학생 대비 위치 파악 가능
  
- 개인별 부족한 학습 요인을 확인하여 맞춤형 개선 전략 수립 가능
  
- 데이터 기반 피드백을 통해 효율적인 학습 관리 및 성적 향상 지원

---

## 팀원 소개
| 이름 | 역할 |
| --- | --- |
| 권선민 | K-Means 군집화 |
| 박주현 | Logistic Regression 모델링, 등급 예측 기능 구현 |
| 송다현 | Random Forest 모델링 |
* 모델링 외 데이터 수집, EDA, 전처리는 팀 전체가 함께 진행했습니다.
