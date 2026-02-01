# GES Benchmark Experiments

NeurIPS 2026 D&B 논문을 위한 벤치마크 실험 코드.

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
# 모든 모델 벤치마크
python benchmark_models.py --task all --n-samples 100

# 이미지 분류 모델만
python benchmark_models.py --task image --n-samples 1000

# 텍스트 분류 모델만
python benchmark_models.py --task text --n-samples 1000

# CPU에서 실행
python benchmark_models.py --task all --device cpu
```

## 벤치마크 모델

### Image Classification (ImageNet)
| Model | Parameters | Accuracy |
|-------|------------|----------|
| ResNet-50 | 25M | 76.1% |
| EfficientNet-B0 | 5.3M | 77.1% |
| MobileNetV3-Large | 5.4M | 75.2% |
| MobileNetV3-Small | 2.5M | 67.5% |

### Text Classification (GLUE)
| Model | Parameters | Accuracy |
|-------|------------|----------|
| BERT-base | 110M | 88.5% |
| DistilBERT | 66M | 87.2% |
| TinyBERT | 14.5M | 84.5% |
| ALBERT-base | 12M | 86.3% |

### Language Generation
| Model | Parameters |
|-------|------------|
| GPT-2 Small | 124M |
| GPT-2 Medium | 355M |

## 결과 출력

결과는 `results/` 폴더에 저장됩니다:
- `benchmark_results_YYYYMMDD_HHMMSS.json` - 전체 결과
- `results_table_YYYYMMDD_HHMMSS.tex` - LaTeX 테이블
- `results_table_YYYYMMDD_HHMMSS.md` - Markdown 테이블

## 논문 연결

- 논문 작업: `논문 작성/논문_GES/1.(이상금) NeurIPS 2026 D&B/`
- GitHub: https://github.com/ecoailab/ecoai-efficiency
