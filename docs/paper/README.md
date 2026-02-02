# GES Paper - NeurIPS 2026 D&B

> **Note**: 논문 작업은 별도 폴더에서 진행됩니다.

## 논문 위치

```
SynologyDrive/논문 작성/논문_GES/1.(이상금) NeurIPS 2026 D&B/
├── paper/
│   ├── main.tex          # 최신 논문 (~1000 lines, v5.1)
│   ├── references.bib
│   └── figures/
├── simulations/
│   ├── comprehensive_benchmark.py  # 74-model benchmark
│   ├── edge_benchmark.py           # 11 edge devices
│   └── statistical_validation.py   # Bootstrap + significance
└── leaderboard/
    └── index.html        # Interactive leaderboard
```

## 논문 현황 (v5.1)

| Metric | Value |
|--------|-------|
| Score | 9.6/10 |
| Best Paper Prob. | 58-63% |
| Models | 85 (74 datacenter + 11 edge) |
| Hardware | 16 platforms |
| Countries | 15 |

## 주요 발견

1. **Efficiency Cliff**: 100M params 이상에서 GES 급격히 하락
2. **Green Paradox**: 정확도 향상 시 super-linear 효율 비용
3. **Compression Sweet Spot**: 압축 모델이 최적 효율-정확도 균형

## 논문 인용

```bibtex
@inproceedings{lee2026ges,
  title = {GES: A Unified Metric and Benchmark for AI Energy Efficiency},
  author = {Lee, Sangkeum},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year = {2026}
}
```

---

*EcoAI Lab, Hanbat National University*
