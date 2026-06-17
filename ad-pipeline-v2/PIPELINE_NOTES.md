# Ad Performance Insight Engine — Pipeline (v2)

`Ads_Cross_Platform.ipynb`의 실제 코드를 그대로 반영해 모듈화한 버전입니다.
컬럼명, 피처, 모델 하이퍼파라미터, 데이터 리키지 체크, anomaly 로직이
노트북과 정확히 일치합니다.

## 노트북 → 모듈 매핑

| 노트북 셀 | 모듈 | 내용 |
|---|---|---|
| [5]-[6] | `src/data_loader.py` | kagglehub 다운로드, 스키마 검증 |
| [10] | `src/features.py` | conversion_rate, ctr_cpc_ratio, log_impressions, cost_per_impression, high_roas 라벨, 카테고리 인코딩 |
| [12] | `src/train.py` → `benchmark_5_models()` | 5-fold CV로 5개 모델 비교 |
| [14] | `src/train.py` → `train_logistic_regression()` | LR 상세 평가 (ROC-AUC 0.8777) |
| [15] | `src/train.py` → `check_data_leakage()` | CPA 제거 후 재학습, ΔAUC 확인 |
| [18]-[19] | `src/train.py` → `tune_xgboost_optuna()` | Optuna 100 trials 하이퍼파라미터 튜닝 |
| [23] | `src/explain.py` | LogReg(LinearExplainer) + XGBoost(TreeExplainer) SHAP |
| [25] | `src/anomaly.py` | Z-score + Isolation Forest |
| [29] | `src/anomaly.py --exclude-tiktok` | TikTok 제외 재실행 |

## 실제 데이터 스키마

```
platform, campaign_type, industry, country   (카테고리)
impressions, clicks, CTR, CPC, ad_spend,
conversions, CPA, revenue, ROAS                (수치)
```

타겟: `high_roas = 1 if ROAS > median(ROAS) else 0` — median은 데이터 로드 시
계산되므로 실행할 때마다 정확한 값이 로그에 출력됩니다 (노트북 기준 4.295).

## 실행 방법 (Docker — 권장)

Mac에서 libomp/OpenMP 충돌 문제를 겪었다면 Docker가 훨�르 안정적입니다.
컨테이너 안은 Linux라 `libgomp1`만 설치하면 XGBoost가 바로 동작합니다.

```bash
# 1. 이미지 빌드
docker compose build

# 2. 파이프라인 단계들을 컨테이너 안에서 순서대로 실행
#    (run --rm: 실행 후 컨테이너 자동 삭제, api 서비스는 그대로 안 띄움)
docker compose run --rm api python src/data_loader.py
docker compose run --rm api python src/features.py
docker compose run --rm api python src/train.py
docker compose run --rm api python src/explain.py
docker compose run --rm api python src/anomaly.py

# 3. API 서버 띄우기
docker compose up
```

`data/`, `models/` 폴더는 `docker-compose.yml`에서 볼륨으로 마운트되어 있어서,
컨테이너 안에서 생성된 CSV/pkl 파일이 네 Mac 로컬 폴더에도 그대로 보입니다.

Kaggle 인증은 `~/.kaggle/kaggle.json`이 자동으로 컨테이너에 마운트되도록
설정해놨습니다 — 로컬에 그 파일이 있으면 따로 설정할 게 없습니다.

API 확인:
```bash
curl http://localhost:8000/health
```

컨테이너 중지:
```bash
docker compose down
```

## 실행 방법 (로컬 — 대안)

```bash
pip install -r requirements.txt

# Kaggle 인증 설정 (~/.kaggle/kaggle.json) 후
python src/data_loader.py        # kagglehub로 자동 다운로드
python src/features.py
python src/train.py              # 5개 모델 벤치마크 + Optuna 100 trials (시간 소요)
python src/explain.py
python src/anomaly.py
python src/anomaly.py --exclude-tiktok

# API 서버
uvicorn api.main:app --reload
```

## API 사용 예시

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

`service.py`가 `conversion_rate`, `ctr_cpc_ratio` 등 파생 피처와 카테고리
인코딩을 서버 측에서 자동 계산하므로, 클라이언트는 원본 캠페인 지표만
보내면 됩니다.

## 검증 상태

5단계 파이프라인 + API 2개 엔드포인트 모두 실제 스키마와 동일한 구조의
synthetic 데이터(1,800행, 균형 클래스 900/900)로 end-to-end 실행 검증
완료. 실제 Kaggle 데이터를 넣으면 노트북과 동일한 결과(LogReg ROC-AUC
0.8777, XGBoost 0.8633, 19개 high-confidence anomaly)가 나와야 합니다.

## 알아둘 것

- `train.py`의 Optuna 튜닝은 100 trials 기본값이라 로컬에서 몇 분 걸릴 수
  있습니다. 빠른 테스트 시 `tune_xgboost_optuna(X_train, y_train, n_trials=10)`
  처럼 줄여서 실행 가능합니다.
- `anomaly.py`는 인코딩 전 원본 데이터(`clean_ads.csv`)를 사용합니다 —
  `features.csv`(인코딩된 테이블)가 아닙니다. 노트북에서도 별도 데이터프레임
  (`df_anomaly = df.copy()`)으로 처리했기 때문입니다.
- Kaggle API 인증이 필요합니다 (`kaggle.json`을 `~/.kaggle/`에 배치).
