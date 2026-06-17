# Ad Performance Insight Engine — Pipeline

`Ads_Cross_Platform.ipynb`의 분석 단계를 독립 실행 가능한 파이썬 모듈로
분리하고, Service Layer 패턴의 FastAPI로 서빙하는 버전입니다.
컬럼명, 피처, 모델 하이퍼파라미터, 데이터 리키지 체크, anomaly 로직은
원본 노트북과 정확히 일치합니다.

## 왜 노트북을 모듈로 나눴나

노트북은 위에서 아래로 셀을 다 실행해야 재현되는 구조라 디버깅과 자동화가
어렵습니다. 이 구조는 각 단계가 **입력 파일 → 처리 → 출력 파일**로 분리되어
있어서, 한 단계만 독립적으로 재실행하거나 중간 결과를 파일로 확인할 수
있고, 모델 학습과 서빙이 완전히 분리됩니다.

```
data/raw/*.csv
      │
      ▼
[1] src/data_loader.py   kagglehub로 다운로드, 스키마 검증
      │
      ▼  data/processed/clean_ads.csv
[2] src/features.py      CTR/CPA 파생 피처, high_roas 라벨, 카테고리 인코딩
      │
      ▼  data/processed/features.csv
[3] src/train.py         5개 모델 벤치마크 → 리키지 체크 → Optuna 튜닝 → 베스트 저장
      │
      ▼  models/best_model.pkl, scaler.pkl, label_encoders.pkl
[4] src/explain.py       SHAP 피처 중요도
[5] src/anomaly.py       Z-score + Isolation Forest 교차검증
      │
      ▼
[API] api/main.py        /health, /predict 서빙
```

## 노트북 → 모듈 매핑

| 노트북 셀 | 모듈                                           | 내용                                                                                                  |
| --------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [5]-[6]   | `src/data_loader.py`                           | kagglehub 다운로드, 스키마 검증                                                                       |
| [10]      | `src/features.py`                              | conversion_rate, ctr_cpc_ratio, log_impressions, cost_per_impression, high_roas 라벨, 카테고리 인코딩 |
| [12]      | `src/train.py` → `benchmark_5_models()`        | 5-fold CV로 5개 모델 비교                                                                             |
| [14]      | `src/train.py` → `train_logistic_regression()` | LR 상세 평가                                                                                          |
| [15]      | `src/train.py` → `check_data_leakage()`        | CPA 제거 후 재학습, ΔAUC 확인                                                                         |
| [18]-[19] | `src/train.py` → `tune_xgboost_optuna()`       | Optuna 100 trials 하이퍼파라미터 튜닝                                                                 |
| [23]      | `src/explain.py`                               | LogReg(LinearExplainer) + XGBoost(TreeExplainer) SHAP                                                 |
| [25]      | `src/anomaly.py`                               | Z-score + Isolation Forest                                                                            |
| [29]      | `src/anomaly.py --exclude-tiktok`              | TikTok 제외 재실행                                                                                    |

## 데이터 스키마

```
platform, campaign_type, industry, country        (카테고리)
impressions, clicks, CTR, CPC, ad_spend,
conversions, CPA, revenue, ROAS                     (수치)
```

타겟: `high_roas = 1 if ROAS > median(ROAS) else 0`. median은 실행 시점에
계산되므로 로그에 정확한 값이 출력됩니다 (실측 4.295).

## API 구조 — Service Layer 패턴

```
api/
├── main.py            라우팅만 (/health, /predict) — 비즈니스 로직 없음
├── service.py           원본 캠페인 지표 → 파생 피처 계산 → 모델 호출
├── model_loader.py       모델/스케일러/인코더 추상화 — sklearn은 여기만 알고 있음
├── schemas.py            Pydantic 요청/응답 스키마
└── logging_config.py     구조화된 JSON 로깅
```

`main.py`는 모델이 LogisticRegression에서 다른 모델로 바뀌어도 코드가
바뀌지 않습니다. `model_loader.py`만 교체하면 됩니다.

### API 사용 예시

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "impressions": 100000, "clicks": 2500, "CTR": 0.025, "CPC": 1.2,
    "ad_spend": 3000.0, "conversions": 80, "CPA": 37.5,
    "platform": "Meta Ads", "campaign_type": "Video",
    "industry": "Retail", "country": "US"
  }'
```

## 실행 방법 — Docker (권장)

로컬 Mac에서 XGBoost가 `libomp.dylib` 충돌로 실행되지 않는 문제를 겪었기
때문에 Docker로 전환했습니다. 컨테이너 안은 Linux라 `libgomp1`만 설치하면
바로 동작합니다.

```bash
docker compose build

docker compose run --rm api python src/data_loader.py
docker compose run --rm api python src/features.py
docker compose run --rm api python src/train.py
docker compose run --rm api python src/explain.py
docker compose run --rm api python src/anomaly.py

docker compose up
curl http://localhost:8000/health
```

`data/`, `models/`, `benchmarks/`는 볼륨으로 마운트되어 있어 컨테이너 안에서
생성된 파일이 로컬에도 그대로 보입니다. `~/.kaggle/kaggle.json`이 있으면
Kaggle 인증이 자동으로 적용됩니다 (읽기 전용으로 마운트하면 kagglehub가
로그 폴더를 쓰지 못해 에러가 나므로 쓰기 가능하게 마운트해야 합니다).

코드를 수정한 뒤에는 `docker compose build`를 다시 실행해야 컨테이너 안에
반영됩니다 — `COPY . .`는 빌드 시점에만 실행되기 때문입니다.

## 학습 단계 성능 — 순차 vs 병렬 실행

`benchmark_5_models()`(노트북 셀 [12])는 원래 5개 모델 × 5-fold CV를
**순차로**, 게다가 accuracy와 ROC-AUC를 따로 두 번(`cross_val_score`를
두 번 호출) 계산하고 있었습니다. 8코어 환경에서 이를 `cross_validate`
한 번 호출 + `n_jobs=-1`로 병렬화하고 실측했습니다.

| 실행 방식                       | 소요 시간              |
| ------------------------------- | ---------------------- |
| Sequential (`n_jobs=1`)         | 815.58s (약 13분 36초) |
| Parallel (`n_jobs=-1`, 8 cores) | 19.40s                 |
| **Speedup**                     | **42.04x**             |

측정 환경: 8-core 컨테이너, 1,800행 데이터셋, 동일한 5개 모델·5-fold 구성.
재현 가능한 측정 스크립트는 `benchmark_parallelism.py`이며, 결과는
`benchmarks/parallelism_results.csv`(누적 기록)와
`benchmarks/latest_summary.json`(최신 요약)에 저장됩니다.

42배는 8코어 병렬화 효과(이론상 최대 ~8배)만으로는 설명되지 않는데,
중복 CV 호출 제거(2배)와 sequential 모드에서 XGBoost가 내부적으로
스레드 경쟁을 일으켰을 가능성이 함께 작용한 것으로 보입니다.

```bash
docker compose run --rm api python benchmark_parallelism.py
```

## 멘토 리뷰용 질문

1. Service Layer로 분리한 게 실무 패턴이랑 맞는지?
2. 모델 버전 관리(`MODEL_VERSION` 문자열 하드코딩)를 MLflow Model Registry
   같은 도구로 정교화해야 하는지?
3. 42배 speedup의 원인을 좀 더 분리해서(코어 병렬화 vs 중복 계산 제거 vs
   스레드 경쟁) 측정하는 게 의미 있을지?
4. 다음 단계로 Prometheus `/metrics` 노출 + Grafana 대시보드를 붙이려는데,
   이 시점이 적절한지, 아니면 더 먼저 해야 할 게 있는지?

## 다음 단계 (보류)

- [ ] Prometheus `/metrics` 엔드포인트
- [ ] Grafana 대시보드 (예측 분포, latency, 이상치 카운트)
- [ ] sequential/parallel speedup의 원인별 분리 측정
- [ ] Optuna 튜닝도 동일한 방식으로 벤치마크

## 알아둘 것

- Kaggle API 인증이 필요합니다 (`kaggle.json`을 `~/.kaggle/`에 배치).
- `train.py`의 Optuna 튜닝은 100 trials가 기본값이라 시간이 걸립니다.
  빠른 테스트 시 `tune_xgboost_optuna(X_train, y_train, n_trials=10)`처럼
  줄여서 실행 가능합니다.
- `anomaly.py`는 인코딩 전 원본 데이터(`clean_ads.csv`)를 사용합니다 —
  노트북에서도 별도 데이터프레임(`df_anomaly = df.copy()`)으로 처리했기
  때문입니다.
