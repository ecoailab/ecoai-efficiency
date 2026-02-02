# GES Benchmark Experiments

NeurIPS 2026 D&B 논문을 위한 벤치마크 실험 코드.

## 실험 스크립트

| 파일 | 설명 | 모델 수 |
|------|------|--------|
| `benchmark_models.py` | 기본 벤치마크 | 10개 |
| `benchmark_extended.py` | 확장 벤치마크 | 30개+ |
| `quick_demo.py` | 빠른 데모 | 4개 |

## 설치

```bash
pip install -r requirements.txt
```

## 실행

### 확장 벤치마크 (권장)

```bash
# 모든 모델 벤치마크 (30+ models)
python benchmark_extended.py --task all --n-samples 100

# 빠른 테스트
python benchmark_extended.py --task all --quick

# 특정 태스크만
python benchmark_extended.py --task image --n-samples 1000
python benchmark_extended.py --task text --n-samples 1000
python benchmark_extended.py --task generation --n-samples 50
```

### 기본 벤치마크

```bash
python benchmark_models.py --task all --n-samples 100
```

## 벤치마크 모델 (v1.0)

### Image Classification (12 models)

| Model | Parameters | ImageNet Acc |
|-------|------------|--------------|
| ResNet-18/34/50/101 | 11.7M-44.5M | 69.8%-77.4% |
| EfficientNet-B0/B1/B2 | 5.3M-9.2M | 77.1%-79.8% |
| MobileNetV2/V3-S/V3-L | 2.5M-5.4M | 67.5%-75.2% |
| ViT-B/16, Swin-T | 28M-86M | 81.1%-81.3% |

### Text Classification (8 models)

| Model | Parameters | GLUE Acc |
|-------|------------|----------|
| BERT-base/large | 110M-340M | 88.5%-90.9% |
| DistilBERT | 66M | 87.2% |
| TinyBERT | 14.5M | 84.5% |
| ALBERT-base | 12M | 86.3% |
| RoBERTa-base | 125M | 90.2% |
| ELECTRA-small | 14M | 85.1% |
| DeBERTa-v3-small | 44M | 88.3% |

### Language Generation (6 models)

| Model | Parameters |
|-------|------------|
| GPT-2 Small/Medium/Large | 124M-774M |
| GPT-Neo-125M | 125M |
| Phi-1.5 | 1.3B |
| TinyLlama-1.1B | 1.1B |

### Object Detection (4 models)

| Model | Parameters | COCO mAP |
|-------|------------|----------|
| Faster R-CNN | 41.8M | 37.0 |
| RetinaNet | 34.0M | 36.4 |
| SSD300 | 35.6M | 25.1 |
| FCOS | 32.3M | 39.2 |

## 결과 출력

결과는 `results/` 폴더에 저장됩니다:

```
results/
├── extended_benchmark_YYYYMMDD_HHMMSS.json  # 전체 결과
├── extended_benchmark_YYYYMMDD_HHMMSS.md    # 요약 리포트
├── results_table_YYYYMMDD_HHMMSS.tex        # LaTeX 테이블
└── results_table_YYYYMMDD_HHMMSS.md         # Markdown 테이블
```

## GES 등급 기준 (v1.0)

74-model benchmark population percentile 기반:

| Grade | GES Score | Percentile |
|-------|-----------|------------|
| A+ | ≥ 3,265,200 | Top 10% |
| A | ≥ 1,306,469 | Top 25% |
| B | ≥ 512,892 | Top 50% |
| C | ≥ 187,135 | Top 75% |
| D | < 187,135 | Bottom 25% |

## 논문 연결

- **논문 폴더**: `논문 작성/논문_GES/1.(이상금) NeurIPS 2026 D&B/`
- **GitHub**: https://github.com/ecoailab/ecoai-efficiency
- **논문**: NeurIPS 2026 Datasets & Benchmarks Track

---

*EcoAI Lab, Hanbat National University*
