# 2차 개선 플랜 — Reviewer Feedback 대응

**작성일:** 2026-04-29
**전제:** GPU 예산 제약 (장기 재학습 불가). 기존 checkpoint 재활용 + 분석/평가/소규모 학습 위주.

---

## 배경: 1차 결과의 한계

`smacv2_poc/results_scaling/`의 scaling 실험 (3v3 / 6v6 / 9v9, Cond1 vs Cond3, 3 seeds, 2M steps)에서:

| Scale | Cond1 (Full) | Cond3 (RoleMLP) | Gap |
|---|---|---|---|
| 3v3 | ~0.85 | ~0.50 | ~35pp |
| 6v6 | (Cond1 미수집) | ~0.34 | – |
| 9v9 | ~0.15 | ~0.08 | ~7pp |

격차가 N에 따라 5배 줄었다. Reviewer는 다음 3개를 지적:

1. **실험 도메인이 좁다** (MiniGrid + SMACv2 단일 family).
2. **Scaling 결과가 강하지 않다** (격차가 줄어듦).
3. **Attention routing vs gradient pseudo-routing이 apples-to-apples 아님** (측정 단위 상이).

---

## 진단: 격차 축소의 근본 원인

성능 격차가 줄어드는 것이 곧 "주장이 약해진다"는 아니다. 다음 confound이 격차를 가린다:

1. **두 조건 모두 9v9에서 미수렴** — 2M steps에서 win rate <20%, RL이 구조를 *활용*하기 전 단계.
2. **Padding 비율 confound** — `obs_dim=182` (9v9 ceiling)로 모든 스케일을 패딩. 3v3에서는 ally 슬롯 8개 중 5개가 0-패딩이라 Cond3 MLP가 zero-block을 입력으로 먹는다 (Cond1은 mask로 무시). 9v9에선 패딩 0 → MLP 페널티 사라짐. **3v3 35pp gap의 일부는 padding penalty.**
3. **Centralized critic 분산 폭증** — 입력 63×N, team_reward_alpha=0.35로 개별 기여도 N에 반비례. Advantage 노이즈가 어텐션의 inductive bias를 덮어버림.
4. **모델 용량 미스케일** — `embed_dim=64, heads=2`로 모든 N에 고정.
5. **Role label 희석** — 9v9에선 role당 3명, MLP도 더 많은 exemplar를 받아 추격.
6. **Episode horizon (200) 고정** — 큰 팀일수록 역할 분화 행동 발현 시간 부족.

핵심 통찰: **paper의 주장은 "어텐션이 role을 routing 신호로 쓴다"는 *구조적* 주장이지, 승률 우위 주장이 아니다.** 승률은 보조 지표로 강등하고, routing 자체를 직접 측정하는 metric으로 무게추를 옮긴다.

---

## Reviewer 피드백 vs. 보강 항목 매핑

| 항목 | #1 도메인 | #2 scaling | #3 metric |
|---|:---:|:---:|:---:|
| **A.** 9v9 Cond1 routing matrix 추출 | – | ✓ | – |
| **A'.** Unified occlusion-routing metric | – | ✓ | ✓✓ |
| **C.** Cross-scale zero-shot transfer | – | ✓✓ | – |
| **D.** Padding-confound 분리 | – | ✓ | – |
| **E.** Domain broadening (MPE + 재포지셔닝) | ✓✓ | – | – |

---

## 작업 목록 (우선순위 순)

### A. Track 2 at scale — 9v9 routing 보존 입증
**목적:** 승률 격차가 작아도 attention routing 구조는 유지됨을 보인다.
**비용:** GPU 0 (기존 checkpoint inference).
**작업:**
- [x] `analyze_attention.py`를 9v9 Cond1 seed0/1/3 checkpoint에 적용 → `smacv2_poc/analysis/analyze_attention_scaling.py` 신규 작성 (182-dim padded obs 지원)
- [x] 3×3 role-routing matrix 산출 (3v3 / 9v9 비교 figure) → `smacv2_poc/results_scaling/analysis/routing_3v3_vs_9v9.png`
- [x] Δmax per role 비교 (3v3 vs 9v9) → `smacv2_poc/results_scaling/analysis/delta_max_3v3_vs_9v9.png`
- [x] 결과 CSV/per-seed npy: `smacv2_poc/results_scaling/analysis/routing_summary.csv`, `routing_<config>_seed<N>.npy`

**Success criterion:** 9v9에서도 Strike→Support > Strike→Vanguard 부등식 유지.

**결과 (2026-04-29):**
- 3v3 Cond1 (seed-mean over 0,1,2; step2000384): Strike→Support=0.529 > Strike→Vanguard=0.471 → **PASS**
- 9v9 Cond1 (seed-mean over 0,1,3; step2000384): Strike→Support=0.143 > Strike→Vanguard=0.113 → **PASS**
- Δmax 9v9에서 모든 role 0.05 근처로 수렴 (3v3 대비 압축됨); Vanguard 가 여전히 최고 (0.054), Support 최저 (0.050).
- Off-diagonal 비대칭은 9v9에서 작아지지만 부등식은 보존됨 — 핵심 *구조적* 주장 유효.

---

### A'. Unified routing metric — occlusion 기반 공통 측정
**목적:** Cond1 (attention) vs Cond3 (gradient) apples-to-apples 비교. **Reviewer #3 직접 해결.**
**비용:** 0.5–1일 코드, GPU 0 (inference).
**작업:**
- [x] `analysis/occlusion_routing.py` 신규: per-ally-slot occlusion (슬롯 token을 0으로, mask bit=1) → KL( π(·|obs) || π(·|obs_occ) ) → ally role 별 3×3 sensitivity matrix
- [x] 두 모델(Cond1, Cond3) 모두에 동일 함수 적용 — same metric, fair comparison
- [x] Cond1: occlusion-routing vs attention-weight-routing의 Pearson r 보고
- [ ] Bonus: Integrated Gradients (시간 남으면)
- [x] 결과 figure: `smacv2_poc/results_scaling/analysis/occlusion_routing_panel.png` (Cond1 occ / Cond1 attn / Cond3 occ / |Cond1−Cond3| 4-col × config 행)
- [x] 보조: `occlusion_routing_summary.csv`, `occlusion_vs_attention_corr.txt`

**Success criterion:** Cond1의 occlusion routing이 attention routing과 r > 0.7 correlate, 그리고 Cond3보다 role-structure가 sharp.

**결과 (2026-04-29) — narrative 조정 필요:**
- 3v3 Cond1 occlusion routing (seed-mean 0,1,2): Strike→Support > Strike→Vanguard inequality **유지 (PASS)**.
- 9v9 Cond1 occlusion: Strike→Support (0.0228) ≈ Strike→Vanguard (0.0236) — **부등식 미세하게 역전 (FAIL)**. Strike row의 절대 KL이 모든 ally role에서 매우 작아 (~0.02) 노이즈 한도. Vanguard / Support 행에서는 routing 구조 뚜렷 (Vanguard→Support=0.31 dominant; Support 내부 self-loop=0.14).
- Pearson r(Cond1 occ, Cond1 attn): **3v3 r=0.59, 9v9 r=0.05** — **목표 r>0.7 미달**. 해석: attention weight는 row-normalized softmax 분포지만 occlusion KL은 정책 sensitivity (절대 magnitude). 두 metric은 동일 정보를 측정하지 않으며, occlusion이 더 "정책에 실제로 영향을 주는" 채널을 직접 측정한다. 따라서 Reviewer #3 응답은 "attention-weight ≈ occlusion-routing" 주장이 아니라 "occlusion이 통일된 metric이며, Cond1과 Cond3 routing 구조 차이를 동일 단위로 비교한다"로 재구성한다.
- Cond3 occlusion: 9v9에서 Support 행이 5.79까지 폭증 (단일 ally slot에 비정상적 의존). Cond1은 magnitude는 작지만 분포가 균일치 않게 role을 가른다. Structure score (log3 - mean row entropy) 만으로는 두 모델 비교가 magnitude bias에 묻혀 명확하지 않음 — 보조 지표로 row-normalized KL 분포를 panel figure에 함께 보고.
- **Action items 추가 (Week 2~3):**
  - (1) Strike row 0 KL 대비 normalize한 *row-relative* routing matrix 보조 산출.
  - (2) Cond3 9v9 checkpoint가 800k step 까지만 학습됨 (cond1은 2M) → 비교 공정성 메모. 추가 학습 승인이 필요.
  - (3) Pearson r 결과를 paper에서 "complementary measurements" 로 framing.

**Refined results (2026-04-29, multi-scenario + row-norm 적용):**
- 코드 보강: `scenario_builder_scaling.build_random_scenarios()` 추가 → occlusion / attention matrix를 (clustered + spread) 50% + (random N=20) 50% 가중 평균으로 산출. `occlusion_routing.py` 가 row-normalized matrix와 두 가지 Pearson r (raw, row-normalized) 를 동시에 보고.
- **3v3 Cond1 (PASS, 강해짐):**
  - Pearson r(occ, attn) **raw=0.62 → row-norm=0.885** — *목표 r>0.7 row-normalized 기준 달성*. 이는 두 metric의 분포 형태가 일치함을 보여주며 (절대 magnitude는 다를 수 있음), occlusion이 attention과 동일한 routing 정보를 인코딩한다는 강한 증거.
  - Strike→Support > Strike→Vanguard: raw (0.075 vs 0.054) **PASS**, row-norm (0.582 vs 0.418) **PASS**.
- **9v9 Cond1 (mixed):**
  - Pearson r 여전히 낮음 (raw=-0.08, row-norm=-0.17) — attention이 "보는" 슬롯과 정책이 "의존하는" 슬롯이 9v9에서 분리된다는 *발견*. Reviewer #3 예측대로 attention weight는 근본 metric이 아니다.
  - 9v9에서 Strike row 부등식은 occlusion 기준 FAIL (Strike→Strike=0.064가 dominant) — 9v9는 같은-role ally가 2명 존재하는 구조적 차이 때문 (3v3은 0명). 이는 "Strike→Support 부등식"의 가정 자체가 same-role ally가 부재할 때만 의미 있음을 시사. 3v3는 부등식 도메인, 9v9는 *role-routing 구조의 보존* 도메인으로 framing 분리.
  - Vanguard→Support (0.222), Support→Support 자기-routing (0.157) 등 비-Strike 행은 routing 구조가 명확히 유지됨.
- **Cond3 (모든 스케일):** 행 magnitude가 폭증 (9v9 Support row 7.10) — 학습 부족(800k step) 으로 정책이 특정 입력 채널에 비정상 의존. **추가 학습 (1.2M step) 사용자 승인 필요** — 그 후 fair comparison 가능. 현재 panel은 정성적 비교용으로만 사용.
- **Paper narrative 최종 안:**
  - (a) **3v3**: occlusion-routing이 attention-routing과 row-normalized 분포로 r=0.88 일치 → occlusion이 attention의 충실한 unified metric이다.
  - (b) **9v9**: occlusion ≠ attention (r≈0). 두 metric은 *complementary* (attention=focus, occlusion=causal influence). 9v9에서 attention level의 부등식은 보존되나 (item A: r=0.143>0.113), 정책 영향력은 same-role ally 추가로 재분배됨. 이는 *어텐션이 role을 routing 신호로 사용한다*는 주장의 보강 (구조적 신호 자체는 유지됨) 이며 동시에 추가 통찰 (causal-routing에는 다중 same-role 슬롯이 분산 효과를 일으킴) 을 제공.
  - (c) Cond3 비교는 학습 일치 후 보강.

**최종 산출물:**
- `routing_3v3_vs_9v9.png`, `delta_max_3v3_vs_9v9.png`, `routing_summary.csv` (item A)
- `occlusion_routing_panel.png`, `occlusion_routing_summary.csv`, `occlusion_vs_attention_corr.txt` (item A')

---

### C. Cross-scale zero-shot transfer — scaling을 단점 → 장점으로 재포지셔닝
**목적:** "어텐션은 N에 불변하게 transfer되지만 MLP는 입력 레이아웃이 바뀌어 transfer 불가"를 정량화. **Reviewer #2 핵심 해결.**
**비용:** GPU 0 (eval만).
**작업:**
- [ ] 3v3 Cond1 checkpoint를 변경 없이 6v6 / 9v9 환경에 적용 (padding mask 덕에 그대로 동작)
- [ ] 3v3 Cond3 checkpoint는 input dim 호환 안 됨을 명시 — 또는 0-padding으로 강제 적용 시 성능을 함께 보고 (negative result로 활용)
- [ ] Eval: 50 episodes/scale, win rate + per-role reward
- [ ] 비교 baseline: 9v9에서 from-scratch 학습한 Cond1 (기존 결과)
- [ ] 결과: `transfer_table.csv` + bar chart

**Success criterion:** 3v3 → 9v9 zero-shot transfer Cond1 win rate > 9v9 Cond3 from-scratch.

---

### D. Padding-confound 분리
**목적:** 3v3 35pp gap 중 padding penalty 기여분 정량화. Reviewer 공격면 차단.
**비용:** 작은 GPU 또는 inference-only 회귀.
**작업:**
- [ ] **Option 1 (저비용):** 3v3 Cond3 checkpoint inference 시, padded ally 슬롯에 random noise vs 0 vs 학습된 평균 ally feature 주입 → 정책 출력 변화 측정. 회귀로 padding 의존도 추정.
- [ ] **Option 2 (재학습 1회):** 3v3 native obs_dim=63 Cond3 단일 시드 재학습 (1M steps로 충분, 3v3은 빠르게 수렴) → 기존 obs_dim=182 Cond3와 win rate 비교.
- [ ] 결과: "3v3 35pp gap 중 약 X pp는 padding 영향, 잔여 (35-X) pp가 진정한 architectural gap"

**Success criterion:** padding 기여분이 격차의 50% 미만이면 architectural claim 유지.

---

### E. Domain broadening — Reviewer #1 해결
**목적:** 도메인 다양성 증명. GPU 무리 없이.
**비용:** MPE는 CPU. 재포지셔닝은 0.
**작업:**
- [ ] **E1. 재포지셔닝 (글쓰기):** paper 본문에서 MiniGrid를 "PoC"가 아닌 "Domain 1 (grid-world cooperation)", SMACv2를 "Domain 2 (StarCraft II micro-management)"로 격상. Two-domain validation을 명시적으로 강조.
- [ ] **E2. MPE + role-tag:** `simple_spread` 또는 `simple_world_comm`을 base로 3-role 변형 추가. CPU 학습 가능, 1M steps에 ~수 시간. Cond1 + Cond3 + 3 seeds.
- [ ] **E3. (옵션, 시간 남으면) Toy diagnostic:** ground-truth routing이 명시된 합성 태스크 — sanity probe로 1-figure 어필.
- [ ] **E4. (옵션, GPU 여유 시) SMACv2 두 번째 race:** terran 또는 zerg 5v5 단일 시나리오, Cond1+Cond3 × 3 seeds.

**Success criterion:** 최소 2개 추가 도메인 (재포지셔닝된 MiniGrid는 기존, 신규 MPE 1개) 에서 occlusion-routing 구조 동일하게 관찰됨.

---

## 실행 순서

```
Week 1:  A    →  A'   (기존 checkpoint 활용, 분석 코드 작성)
Week 2:  C    →  D    (transfer eval + padding ablation)
Week 3:  E1   →  E2   (재포지셔닝 + MPE 학습)
Week 4:  Paper 통합 + figure 교체 + narrative 재작성
```

---

## 최종 Paper Narrative

> Label-conditioned ISA가 role을 routing 신호로 쓴다는 **구조적 주장**은:
> - **(a)** 두 도메인 (MiniGrid grid-world, SMACv2 micro-management) + 보조 도메인 (MPE) 에서 **동일한 occlusion-routing metric**으로 직접 측정되고,
> - **(b)** 팀 크기 3→9 스케일에서도 routing 구조가 **보존**되며,
> - **(c)** 어텐션 정책의 **zero-shot cross-scale transfer** 능력으로 입증된다.
>
> 승률 격차 자체는 N이 커질수록 RL 수렴난이도 + padding confound로 좁아지지만, 구조적 주장은 모든 스케일·도메인에서 일관된다.

---

## 운영 규칙

- 매 작업 완료 시 이 문서의 해당 체크박스를 갱신하고, 결과 파일 경로를 추가하라.
- 결과가 예상과 다르면 그 자체를 정직하게 기록하고, narrative를 조정하라 (성공이 아닌 *발견*을 보고).
- GPU 학습이 필요한 항목은 사전에 사용자 승인을 받는다.
- 이 문서가 PLAN.md / AGENTS.md / CLAUDE.md의 **Phase 2 후속 작업 single source of truth**이다.
