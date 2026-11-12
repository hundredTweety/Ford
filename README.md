# 🚗 엔진 센서 이상 탐지 (FordA)

> 자동차 엔진 센서 500개의 측정값으로 **정상 / 이상 엔진을 분류**한 시계열 분석 학습 기록 (2024.04)

| 항목 | 내용 |
|---|---|
| 데이터 | [UCR FordA](https://www.timeseriesclassification.com/description.php?Dataset=FordA) — 학습 3,601개 · 테스트 1,320개, 샘플당 센서 값 500개 |
| 목표 | 엔진 소음 측정값으로 이상(-1) / 정상(1) 이진 분류 |
| 사용 기술 | Python · NumPy · pandas · Matplotlib · scikit-learn · XGBoost |

<br>

## 진행 순서

1. **데이터 읽기** — Weka용 `.arff` 파일을 `scipy.io.arff`로 읽어 NumPy 배열로 변환
2. **데이터 확인**
   - 클래스 균형 확인: 정상과 이상 샘플 수가 비슷해서 별도 리샘플링 없이 진행
   - 같은 시점의 정상·이상 샘플 비교, 임의 센서 하나의 시계열 확인(이상치 탐색)
3. **상관 분석** — 센서 번호가 가까울수록 상관이 높고 멀어질수록 낮아지는 패턴 발견
   → "가까운 센서끼리의 관계(지역적 특징)"를 학습하는 **CNN 계열이 유리할 것**이라는 가설
4. **정규화 · 형태 변환** — 라벨을 0/1로 바꾸고, 딥러닝 확장을 고려해 `(N, 500) → (N, 500, 1)` 변환
5. **모델 비교**
   - [`Logistic Regression.ipynb`](Logistic%20Regression.ipynb): scikit-learn 버전과 **NumPy로 직접 구현한 버전**을 함께 학습
   - [`XGBoost.ipynb`](XGBoost.ipynb): GridSearch로 하이퍼파라미터 탐색 후 정확도 · AUC · F1 평가

<br>

## 결과와 배운 점

| 모델 | 테스트 정확도 |
|---|---|
| 로지스틱 회귀 (scikit-learn) | 0.49 |
| 로지스틱 회귀 (NumPy 직접 구현) | 0.49 |

- 선형 모델은 **정확도 49%로 사실상 무작위 수준**이었어요. 500개 센서 값을 독립적인 특성으로 보는 선형 경계로는 이상 신호를 구분할 수 없다는 뜻이에요.
- 이 결과가 상관 분석에서 세운 가설(센서 간 지역적 관계가 중요하다)과 맞아떨어져서, **비선형 모델(XGBoost)과 CNN 계열로 넘어가야 하는 근거**가 됐어요.
- 모델 성능만 보는 대신 **"왜 이 모델이 안 되는지"를 데이터 특성으로 설명하는 연습**을 했어요.

<br>

## 실행

```bash
pip install numpy pandas matplotlib scikit-learn scipy xgboost
```

노트북 안의 `file_path`를 FordA `.arff` 파일이 있는 폴더로 바꾼 뒤 실행해요. 데이터 파일은 용량 때문에 레포에 포함하지 않았어요.
