# GES 논문 계획

## 논문 정보

| 항목 | 내용 |
|------|------|
| **제목** | GES: A Unified Metric for AI Energy Efficiency Measurement |
| **타겟** | NeurIPS 2026 Datasets & Benchmarks Track |
| **예상 마감** | 2026년 5월 |
| **페이지** | 9페이지 + 참고문헌 |
| **형식** | Single-blind |

## 핵심 기여 (Contributions)

1. **GES 메트릭**: 정확도와 에너지를 결합한 단일 효율성 점수
2. **등급 시스템**: A+~D 등급으로 빠른 해석 가능
3. **오픈소스 도구**: ecoai-efficiency (실시간 탄소, SCI 호환, LaTeX 내보내기)
4. **벤치마크**: 12개 인기 모델 효율성 측정

## 논문 구조

### 1. Introduction (1 페이지)
- AI 에너지 소비 증가 문제
- 기존 도구의 한계 (통합 메트릭 없음)
- GES 제안 및 기여점

### 2. Related Work (1 페이지)
- Energy Measurement Tools (CodeCarbon, CarbonTracker, eco2AI)
- Green AI 연구 (Strubell, Schwartz, Patterson, Luccioni)
- Efficiency Metrics (MLPerf, SCI)

### 3. The Green Efficiency Score (2 페이지)
- GES 정의: Accuracy / kWh per 1K inferences
- 등급 시스템 (A+~D)
- SCI와의 관계

### 4. Implementation (1.5 페이지)
- ecoai-efficiency 아키텍처
- 사용법 예시
- 하드웨어 지원

### 5. Experiments (2.5 페이지)
- 실험 설정 (12개 모델, 3개 태스크)
- 결과 테이블
- 주요 발견점

### 6. Limitations & Future Work (0.5 페이지)
### 7. Conclusion (0.5 페이지)

## TODO 리스트

### 실험 (3-4월)
- [ ] ImageNet 모델 측정 (ResNet, EfficientNet, MobileNet)
- [ ] GLUE 벤치마크 측정 (BERT, DistilBERT, TinyBERT, ALBERT)
- [ ] LLM 측정 (GPT-2, Llama-2)
- [ ] 다양한 하드웨어 테스트 (A100, V100, CPU)
- [ ] 지역별 탄소 영향 분석

### 논문 작성 (4월)
- [ ] Abstract 완성
- [ ] Introduction 작성
- [ ] Related Work 작성
- [ ] Method 섹션 작성
- [ ] Experiments 결과 정리
- [ ] Figure/Table 생성

### 제출 준비 (5월)
- [ ] 공저자 확인
- [ ] 코드 정리 및 재현성 확인
- [ ] Supplementary material 준비
- [ ] Camera-ready 준비

## 파일 구조

```
docs/paper/
├── main.tex          # 메인 논문 파일
├── references.bib    # 참고문헌
├── PAPER_PLAN.md     # 이 파일
├── figures/          # 그림 (추후 생성)
│   ├── ges_comparison.pdf
│   └── architecture.pdf
└── supplementary/    # 보조 자료 (추후 생성)
    └── experiments.tex
```

## 주요 참고 논문

1. Strubell et al. (2019) - Energy and Policy Considerations for Deep Learning in NLP
2. Schwartz et al. (2020) - Green AI
3. Patterson et al. (2021) - Carbon Emissions and Large Neural Network Training
4. Luccioni et al. (2023) - Estimating the Carbon Footprint of BLOOM
5. MLPerf Power (2024) - Benchmarking AI Energy Efficiency

## 차별화 포인트

| 기존 도구 | ecoai-efficiency |
|-----------|------------------|
| kWh, gCO2 따로 보고 | **GES 단일 점수** |
| 숫자만 제공 | **A+~D 등급** |
| 정적 탄소 계수 | **실시간 API** |
| 학술 내보내기 없음 | **LaTeX 테이블** |
| SCI 미지원 | **SCI for AI 준수** |

## 예상 리뷰어 질문

1. **왜 GES가 필요한가?** → 기존 도구는 비교 불가능한 여러 숫자 제공
2. **등급 기준은?** → 실험적으로 결정, 업계 표준과 정렬
3. **정확도가 아닌 다른 메트릭은?** → F1, BLEU 등 확장 가능
4. **학습 효율성은?** → Future work로 제안
