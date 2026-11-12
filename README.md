# 🚗 엔진 소음 시계열 이상 분류 (FordA)

> 자동차 엔진 소음을 500개 시점에서 측정한 **단변량 시계열**로 이상 징후 유무를 분류한 시계열 분석 학습 기록 (2024.04)

| 항목 | 내용 |
|---|---|
| 데이터 | [UCR FordA](https://www.timeseriesclassification.com/description.php?Dataset=FordA) — 학습 3,601개 · 테스트 1,320개, 샘플당 엔진 소음 측정값 500개 |
| 목표 | 엔진 소음 측정값으로 이상(-1) / 정상(1) 이진 분류 |
| 사용 기술 | Python · NumPy · pandas · Matplotlib · scikit-learn · XGBoost |

<br>

## 진행 순서

1. **데이터 읽기** — Weka용 `.arff` 파일을 `scipy.io.arff`로 읽어 NumPy 배열로 변환
2. **데이터 확인**
   - 클래스 균형 확인: 정상과 이상 샘플 수가 비슷해서 별도 리샘플링 없이 진행
   - 정상·이상 샘플의 동일 시점 측정값 비교와 개별 시계열 시각화
3. **상관 분석** — 인접한 시점일수록 측정값의 상관이 높은 패턴을 관찰했습니다. 인접 시점의 패턴을 학습하는 1D CNN은 후속 실험 가설입니다.
4. **정규화 · 형태 변환** — 라벨을 0/1로 바꾸고, 딥러닝 확장을 고려해 `(N, 500) → (N, 500, 1)` 변환
5. **모델 실험**
   - [`Logistic Regression.ipynb`](Logistic%20Regression.ipynb): scikit-learn 버전과 **NumPy로 직접 구현한 버전**을 함께 학습
   - [`XGBoost.ipynb`](XGBoost.ipynb): GridSearchCV와 테스트 정확도·ROC-AUC·F1 계산 코드. 저장본의 탐색은 중단됐고, 평가 지표 출력은 저장돼 있지 않습니다.

<br>

## 결과와 배운 점

| 모델 | 테스트 정확도 |
|---|---|
| 로지스틱 회귀 (scikit-learn) | 0.49 |
| 로지스틱 회귀 (NumPy 직접 구현) | 0.49 |

- 두 로지스틱 회귀 구현은 테스트 1,320건에서 반올림 기준 정확도 49%를 기록했습니다. 같은 테스트셋에서 다수 클래스만 예측한 정확도는 약 51.6%(681/1,320)입니다.
- XGBoost와 1D CNN의 성능은 검증을 마친 뒤 같은 테스트셋 기준으로 비교할 수 있습니다.

<br>

## 실행

```bash
pip install numpy pandas matplotlib scikit-learn scipy xgboost
```

노트북 안의 `file_path`를 FordA `.arff` 파일이 있는 폴더로 바꾼 뒤 실행해요. 데이터 파일은 용량 때문에 레포에 포함하지 않았어요.
