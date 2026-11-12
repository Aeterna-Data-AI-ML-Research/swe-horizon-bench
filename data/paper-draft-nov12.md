# The Punica Throughput Penalty and the Smoke-to-Leaderboard Transfer Gap in 4-Bit Quantized SWE Agents

**Authors:** Moran Zavdi (team)  
**Host model:** `gemma-4-31b-it-qat-w4a16-ct` (QAT W4A16)  
**Resources:** Kaggle dataset `moranzavdi/gemma4-soundness-and-punica-bench` (data, `swe_horizon_bench/`, executed notebook).

---

## Abstract

**Problem.** Under a fixed 12 h host wall, smoke-split gains fail to transfer to the leaderboard, and unverified harnesses corrupt Pass@1. **Methodological insight.** Evaluation needs a two-sided soundness gate (empty fails, oracle passes) and joint accounting of accuracy and serving cost: rank-64 LoRA breaks fused 4-bit execution, turning latency into timeouts, while a 150-line read cap hides companion hunks. **Strongest result.** Two-sided gating retains **111/129** tasks; on a matched $N{=}5$ timing core, LoRA matches base Pass@1 (**0.40**) but cuts episode throughput from **20.1 to 11.0 tok/s** (+87 s/task); across 85 usable non-holdout tasks the agent localizes gold files on **84%** of episodes yet resolves **16%**, with post-edit repair succeeding **0/10** times after a failed first edit. **Limitations & practical implication.** Paired E2E samples are small ($N{=}5$ timed; 15 right-censored runs over 10 tasks) and the OOD check is a synthetic probe; within those bounds, the practical recipe under a fixed wall is base inference (`timeout_seconds=240`, **0.18** LB) plus serve-cost-aware trajectory selection (**MS-SFT**). A nine-reader audit finds that our graph reader's gain is lexical and that file retrieval is the localization bottleneck.

---

## 1. Introduction

The host pins `gemma-4-31b-it-qat-w4a16-ct` under a ∼12 h wall at coarse public resolution (∼0.017/task), so single submissions are weak estimators.

On the smoke core ($N{=}5$), matched base-vs-LoRA E2E yields identical Pass@1 (**0.40**) with **+87.2 s** mean wall; across **$N{=}15$** pairs LoRA adds **+90.5 s** per task ($p{<}0.001$) at 33% lower episode throughput (§5.2). Post-localization collapse (**84%→16%**), the 150-line read cap, and **0/10** second-edit repair motivate companion-aware retrieval and compact trajectories (§§5.3–6).

---

## 2. Methods

**Protocol.** Tools: `run_command`, `read_file`, `edit_file`, `write_file`, `get_status`, `submit_patch` (+ optional graph tools). Sampling: `temp=0.2`, `top_p=0.95`, `out=8192`, `think=4096`; budgets 7–8 min, 28 calls, `timeout_seconds`∈{240,300}.

**`submit_patch` (`MEASURED`).** ADK runs `git add -N .` then diffs, so untracked scratch enters the patch; the verifier resets protected tests, **not** scratch.

**Budgets vs grading (`MEASURED`).** `max_tool_calls` gates four tools; exhaustion ≠ empty grade if the working-tree diff passes.

**Statistics.** Per-task analyses exclude the sealed holdout; we report bootstrap 95% CIs (10,000 resamples) and exact Wilcoxon signed-rank $p$.

---

## 3. Soundness Census (`MEASURED`)

### Table 1 — Empty vs oracle on 129 tasks

| Bucket | $n$ | Rule |
| --- | ---: | --- |
| Usable | **111** | empty fails ∧ oracle passes |
| Unusable | **18** | empty-green, oracle-red, or both-green |
| Usable non-holdout | **85** | published per-task ledger |

### Table 1b — Pin recovery (compact)

| Intervention | Effect (`MEASURED`) |
| --- | --- |
| Starlette **0.46.2** + `PYTHONPATH=src` | Unlocks **5** quarantined tasks (**18→13** unusable) |
| Python **3.12** + pytest `--timeout=5` | Recovers **11/11** native-verifier skips |
| Python **3.14** empty-green false solves | Catches `fastapi_14246`, `fastapi_14609` |

![Figure 1](figures/fig1_soundness_and_drift.svg)

---

## 4. Prompt & Tool-Schema Ledger

Table 2 is a submission ledger, not a factorial grid (knobs co-vary). `m`=min, `c`=calls, `T`=tools; all use 8192 output and think 4096 unless noted.

### Table 2 — Official LB ledger (anonymous configuration names)

| ref | Score | Configuration | Key knobs |
| --- | ---: | --- | --- |
| `57076074` | **0.18** | **Config-9T-Compact** | 8m, 240s, 9T, ∼4.7 kB |
| `57538500` | **0.18** | **Config-9T-7m** | 7m, out **6144**, 240s, 9T |
| `57703722` | **0.18** | **Config-6T-Temp1** | 8m, temp **1.0**, 240s, 6T |
| `57665916` | **0.13** | **Config-9T-UncalibratedGraph** | +3 unused graph tools (**−0.04**) |
| `57946188` | **0.13** | **HorizonGuard-9T-7m** | 7m, 240s; +338 B prompt |
| `57848086` | **0.12** | **Baseline-9T-8m** | 8m, 240s, ref pack |
| `57773628` | **0.12** | **Config-6T-Timeout300** | 8m, **300s**, 6T |
| `57425395` | **0.17** | **LoRA-v2-r64** | smoke **0.30** |
| `57594583`/`57570012` | **0.12**/**empty** | **LoRA-v3-r64** | smoke **0.30**; 6m / 8m |
| `58010294` | PENDING | **Config-6T-Strict** | 7m, 6144, 6T |

Each row is one draw (∼0.017/task; gaps <0.03 are noise). All **0.18** winners share `timeout_seconds=240` and 28 calls (`57773628` at 300 s dropped to **0.12**).

---

## 5. Results

### 5.1 Serving mechanism and one throughput framework (`MEASURED`)

At `batch_size=1`, Punica/SGMV adds **∼840 small bf16 launches per token** (7 projections × 2 shrink/expand × 60 layers) outside the fused 4-bit `CompressedTensorsW4A16` path (an analytical count). No server-side decode rate was logged, so every tok/s figure in this paper is **episode throughput**, $R=\sum \text{completion tokens}/\sum \text{episode wall}$, where wall includes prefill, tool execution, and sandbox I/O. Rows share metric and adapter (only the adapter flag changes) and differ only in sample, aggregation, and 480 s censoring.

### Table 3 — Episode throughput, base → LoRA

| Sample | Pairs | Aggregation | Base → LoRA tok/s | Δ | Capped arms (B / L) |
| --- | ---: | --- | :---: | ---: | :---: |
| Smoke split | 10 | pooled | 16.59 → 12.16 | −27% | 3 / 7 |
| Timing core (§5.2) | 5 | pooled | **20.09 → 11.01** | **−45%** | 0 / 0 |
| Both cohorts | 15 | per-episode mean | 18.13 → 12.10 | −33% | 3 / 7 |
| Both arms uncensored | 8 | pooled | 19.74 → 11.95 | −39% | 0 / 0 |

All rows drop 27–45%; the uncensored timing core is the reference. An earlier draft's 13.95→12.10 tok/s paired two LoRA adapters and is retracted. LoRA packs at smoke **0.30** scored **0.17 / 0.12 / empty** on the host (Table 2).

### 5.2 Paired base-vs-LoRA: $N{=}5$ timing + $N{=}15$ with right-censoring (`MEASURED`)

**Timing core ($N{=}5$, Figure 2).** Only the adapter changes: Pass@1 **0.40=0.40** (same two tasks), wall **300.9→388.0 s** (+87.2 s), completion tokens **6043→4272** (−29%), episode throughput **20.09→11.01 tok/s** (−45%), extrapolated wall **10.0→12.9 h**/120. Fewer tokens save ∼88 s; lower throughput adds ∼175 s.

![Figure 2](figures/fig2_paired_e2e_wall_attribution.svg)

**Expanded $N{=}15$ (`PRELIMINARY`).** The smoke-split pairs (10 tasks, same adapter) plus the timing core give 15 pairs over **10 unique tasks** (5 recur). **7/15** pairs have ≥1 arm at the **480 s** cap (**3** on both).

| Estimand (LoRA − Base) | Mean | 95% CI; exact $p$ |
| --- | ---: | --- |
| $\Delta$ wall s ($N{=}15$) | **+90.5** | $[+49,+136]$; $p{<}0.001$ |
| $\Delta$ episode tok/s ($N{=}15$) | **−6.0** | $[−8.1,−4.1]$; $p{<}0.001$ |
| $\Delta$ steps ($N{=}15$ / 10-task cluster) | −5.47 / −5.75 | $[−9.3,−1.4]$; $p{=}0.024$ / $0.043$ |
| $\Delta$ steps (both arms uncensored, $n{=}8$) | −3.5 | $[−8.8,+2.3]$; $p{=}0.28$ |

Pass@1 is Base **4/15** → LoRA **5/15**: LoRA is slower, not less accurate at this $N$. The step reduction is partly mechanical (a capped LoRA episode stops early) and is not significant on uncensored pairs.

### 5.3 Post-localization failure analysis (`MEASURED`)

#### 5.3.1 Post-localization drop and the read→edit gap

On historical smoke+dev ($N{=}44$), gold read **84.1%** drops to gold edit **40.9%**, post-edit pytest **18.2%**, and pass **15.9%** (Figure 3): **19/30** read-gold failures never edit gold, and **11** edit then fail (missed companion hunk **6/11**, scratch in diff **3/11**). Across **138** post-edit-pytest episodes, first-shot PASS is **43/138 (31.2%)**; when `edit_1` fails (**52**), only **10** run a second edit+pytest and **0/10** convert FAIL→PASS (**first-shot edit dominance**). PASS episodes edit all companions before pytest more often (**75.2%** vs **56.8%**, $n{=}27$).

![Figure 3](figures/fig3_post_localization_failure_taxonomy.svg)

#### 5.3.2 Oracle topology and the 150-line horizon law

`read_file` caps at **150 lines / 10k chars** (Figure 4). Across $n{=}85$ functional diffs (**37** single-hunk, **19** single-file multi-hunk, **27** multi-file, **2** metadata-only), Pass@1 drops **27.8% → 11.1% → 5.6%**: single-hunk falls **50%→10%** past line 150; multi-hunk **>150 lines apart** hits **0%** edit/pass; multi-file all-gold edit is **5.6%** despite **89%** symbol coupling.

![Figure 4](figures/fig4_topology_horizon_collapse.svg)

#### 5.3.3 Hunk-budget feasibility and the minimal-sufficient paradox

By hunk count ($n{=}83$; Figure 5), Pass@1 is **25.9%** on 1–3 hunks (**54** tasks) vs **0%** on 4–8 (**16**) and ≥9 (**13**). Among 29 sometimes-solved tasks, **22/29** winners are **minimal-sufficient** (fewer hunks than the gold PR), and **`edit_file` (27/29)**—never `write_file` (**0/29** on gold; **20/29** scratch)—modifies gold.

![Figure 5](figures/fig5_hunk_feasibility_and_n3_taxonomy.svg)

#### 5.3.4 Usable non-holdout hierarchy ($n{=}85$) and native verification

Task-level ever-sound ceiling is **44/85 (51.8%)**. Across **76** failure episodes (**15** scratch-only, **61** package-edit: **17** assertion, **22** exception, **18** collection, **4** timeout), **14/19** right-function edits on never-solved tasks still fail (**11/14** on primary issue assertions), motivating `python3 -c` reproduction checks.

---

## 6. Candidate Remedies — Retrieval Baselines and MS-SFT (`MEASURED`)

### 6.1 AST-HorizonGraph against strong retrieval baselines

**AST-HorizonGraph** builds a symbol graph $G=(V,E_{\mathrm{import}}\cup E_{\mathrm{call}}\cup E_{\mathrm{sibling}}\cup E_{\mathrm{attr}})$ from issue tokens weighted by sub-token IDF, then reads the seed header, distant companions, or coupled files. Nine readers share one budget ($K$ disjoint ≤150-line windows, issue text only) on the 46 multi-hunk/multi-file tasks, scored as the fraction of pre-patch gold-hunk lines covered. **Gold files** is the original protocol (only gold files, gold seed); **full repo** offers every non-test source file without a seed.

| Reader | Gold files @3 / @5 | Full repo @3 / @5 | Repo file recall @3 |
| --- | :---: | :---: | :---: |
| Top-150 (file headers) | .36 / .37 | .11 / .15 | .38 |
| Sequential pages | .39 / .45 | .10 / .13 | .17 |
| BM25 windows [19] | **.55** / .65 | .19 / .28 | .40 |
| Dense, AST chunks (Jina-code [20]) | .48 / .59 | **.27 / .36** | **.51** |
| AST-only (def chunks + BM25) | .45 / .52 | .19 / .24 | .38 |
| Call-graph expansion (2-hop) | .42 / .56 | .13 / .17 | .34 |
| AST-HorizonGraph, uniform weights | **.55** / .67 | .11 / .15 | .21 |
| **AST-HorizonGraph** | .54 / **.69** | .13 / .18 | .22 |
| Oracle window placement | .87 / .92 | .87 / .92 | .84 |

Given the gold files, AST-HorizonGraph beats Sequential (+0.15 at $K{=}3$, $p{<}0.001$) and call-graph expansion (+0.12, $p{=}0.011$) but ties BM25 (−0.003, 95% CI $[−0.10,+0.08]$); uniform weights change nothing (−0.007). Its gain is lexical window placement, not graph structure or IDF. In the full repo, it trails dense retrieval (−0.14, CI $[−0.25,−0.03]$, $p{=}0.018$), which opens 51% vs 22% of gold files, while oracle placement reaches ≥0.87 (exact on 34/46): the gap is *which files* are read. The localizer also infers multi-file topology from pool size, which leaks under gold-only pools. Tests: paired sign-flip; CIs: 10,000 bootstraps.

### 6.2 MS-SFT: serve-cost-aware trajectory *selection* for SFT data

**What MS-SFT is.** A data-curation rule, not a training objective: $\mathcal{J}_{\mathrm{MS}}$ scores rollouts and, per solved task, we keep the single top-scoring one. That set is the corpus for ordinary SFT (unweighted cross-entropy); the score sets no loss weights and is never optimized over model parameters.

$$
\mathcal{J}_{\mathrm{MS}}(\tau)=\mathbb{I}[\mathrm{SoundPass}(\tau)]\cdot\exp\!\Big(-\alpha\tfrac{\mathrm{Tokens}(\tau)}{T_{\max}}-\beta\tfrac{\mathrm{Steps}(\tau)}{N_{\max}}-\gamma\tfrac{\mathrm{Hunks}(\tau)}{\mathrm{Hunks}_{\mathrm{gold}}}-\delta\,\mathbb{I}[\mathrm{Scratch}](\tau)\Big),\qquad \tau^*_t=\arg\max_{\tau\in R_t}\mathcal{J}_{\mathrm{MS}}(\tau)
$$

with $R_t$ the rollouts for task $t$ and $(\alpha,\beta,\gamma,\delta)=(1.5,1.0,0.5,2.0)$, $T_{\max}{=}8192$, $N_{\max}{=}80$ (hand-set, not tuned). Each policy keeps one trajectory per task over the same **43** solved tasks (1,237 non-holdout rollouts), avoiding Simpson’s-paradox over-weighting of easy tasks.

| Selected corpus (1 traj./task, 43 tasks) | Steps | Tokens | Hunks | Proj. LoRA h/120 @4096 | @2048 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform random winner | 28.6 | 4658 | 3.25 | **14.10** | 10.15 |
| Closest to full gold PR | 30.4 | 4970 | 2.23 | **15.05** | 10.83 |
| Fewest steps | 22.6 | 4195 | 2.77 | 12.70 | 9.15 |
| **MS-SFT** ($\tau^*_t$) | **24.1** | **3918** | **2.05** | **11.86** | **8.54** |

**What this table shows, and does not.** Rows describe the *selected data*, not a trained model; wall columns assume a fine-tuned policy reproduces its training token counts at the timing-core LoRA throughput (11.01 tok/s). Since $\mathcal{J}_{\mathrm{MS}}$ penalizes tokens, steps, and hunks, winning those columns is by construction; the finding is that compact sound winners exist for every solved task, enough for a projected wall under 12 h. No MS-SFT adapter has been trained or host-evaluated (§9).

### 6.3 Synthetic OOD probe and toolkit

We release **`swe-horizon-bench`** (`ASTHorizonGraph`, `ms_sft_selector`, `soundness_gate`, CLI) and **SWE-HorizonBench-OOD**: 60 synthetic targets across eight external libraries (0% ID overlap) whose "gold patches" append `# HORIZON_BENCH_FIX` to real `def` lines, verified by marker presence (**11/11** two-sided). Problem statements name the target symbols, so the probe cannot separate graph structure from lexical matching (§6.1) and is not evidence of out-of-distribution localization.

---

## 7. Claim Audit, Discussion, and Limitations

| Claim | Verdict & evidence |
| --- | --- |
| Two-sided gate required; ImportErrors = host-env gaps | **Supported** (**111/129** usable) / **False** (**24/24** = exports/syntax; ceiling **44/85**) |
| LoRA smoke (**0.30**) transfers to LB; LoRA costs no wall time | **Both False** (LB **0.17/0.12/blank**; $N{=}15$ $\Delta$wall **+90.5 s**, $p{<}0.001$) |
| LoRA lowers episode throughput; LoRA takes fewer steps | **Supported** (−33%, $p{<}0.001$) / **Confounded** by censoring (uncensored $p{=}0.28$) |
| Localization ≫ resolution; 150-line cap hides companions; `edit_file` wins | **Supported** (**84.1%→15.9%**; >150-apart **0%**; `edit_file` **27/29** vs `write_file` **0/29**) |
| AST-HorizonGraph gains come from graph structure | **False** (ties BM25 on gold files; trails dense retrieval at repo scale, $p{=}0.018$) |
| Full-gold imitation is serve-cost optimal; unused graph tools are free | **Both False** (worst *projected* **15.05 h**; **−0.04** schema tax; `240s` in all **0.18** winners) |

**Limitations.** Single-draw LB rows cannot resolve <0.03 gaps; $N{=}5$ timed / $N{=}15$ right-censored samples are small; no server-side decode rate was logged; the 8-library probe names target symbols and uses marker checks rather than behavioural tests; and MS-SFT is a data-selection result: no adapter trained on it has been evaluated.

**Reproducibility.** Dataset `moranzavdi/gemma4-soundness-and-punica-bench` ships all tables, figures, `swe_horizon_bench/`, and the executed notebook.

## 8. Related Work

**Repository-level code retrieval.** Iterative retrieve-and-edit pipelines (RepoCoder [1]) and process-level benchmarks (*ContextBench* [2]) show that line-level sufficiency, not file-level match, governs patch success. Our funnel (**84.1%** gold-file read vs **15.9%** pass) isolates the regime where file retrieval succeeds but a **150-line** `read_file` window truncates the companion edit site.

**Graph-based code representations.** Program graphs (AST/CFG/PDG; code2vec [3]) and repository symbol graphs (AutoCodeRover [4], RepoGraph [5]) guide localization over unrestricted contexts. **AST-HorizonGraph** adapts this lineage to a 150-line window budget, but against BM25 [19], dense [20], AST-only, and call-graph baselines its edges add no measurable recall (§6.1). Conversely, Hu et al. [6] show uncalibrated structure tools raise agent cost up to **+28.14%**, matching our **−0.04** LB schema tax (`57665916`).

**Long-context SWE agents.** SWE-agent [7] and Agentless [8] assume generous budgets and frontier APIs. Under a 4-bit host with a 28-call cap and **0/10** second-edit repair, companion-complete *first-shot* edits outweigh multi-turn test-repair loops.

**Efficient inference for PEFT.** Multi-tenant LoRA engines—Punica/SGMV [9] and S-LoRA [10] atop PagedAttention/vLLM [11]—optimize batched adapters, while DoRA [12] decomposes weight updates. At `batch_size=1` on a 4-bit base, unmerged rank-64 LoRA leaves the fused path (§5.1), cutting episode throughput **45%** and breaching the 12 h wall.

**Trajectory selection and outcome-based SFT.** Rejection-sampled self-training (STaR [13], ReST$^{EM}$ [14]) filters rollouts by binary pass. Pass-all filtering over-weights easy tasks and keeps verbose trajectories that time out under LoRA; **MS-SFT** keeps the rejection-sampling recipe (select, then plain SFT) but replaces the binary filter with a per-task cost-aware argmax.

**Benchmark contamination and soundness.** Following SWE-bench [15], LiveCodeBench [16], SWE-Bench+ [17], and test-overfitting audits [18], our two-sided gate quarantines **18/129** broken tasks and catches empty-green false positives.

---

## 9. Priority Experiments

1. Repeat key packs with $N{\ge}3$ (**0.18** winners, Baseline-9T-8m, Config-6T-Timeout300).  
2. Train an adapter on the MS-SFT corpus and measure its tokens, wall, and Pass@1 on the host.  
3. Pair dense file retrieval with in-file companion placement and test it on the host.  
4. Rebuild OOD from real commits, withholding symbol names.

---

## 10. Conclusion

A two-sided soundness gate (**111/129**), serve-cost accounting ($N{=}5$ timing; $N{=}15$ right-censored), and the **84%→16%** localization-to-pass funnel explain why smoke gains fail on the host. Strong baselines place the remaining retrieval gap in file selection, not window geometry; **MS-SFT** targets serve cost.

---

## 11. References

1. F. Zhang et al. (2023). *RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation.* EMNLP, arXiv:2303.12570.
2. H. Li et al. (2026). *ContextBench: A Benchmark for Context Retrieval in Coding Agents.* arXiv:2602.05892.
3. U. Alon et al. (2019). *code2vec: Learning Distributed Representations of Code.* POPL, PACMPL 3:40, arXiv:1803.09473.
4. Y. Zhang et al. (2024). *AutoCodeRover: Autonomous Program Improvement.* ISSTA, arXiv:2404.05427.
5. S. Ouyang et al. (2024). *RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph.* arXiv:2410.14684.
6. Y. Hu et al. (2026). *Analyzing and Mitigating Cost-Inefficient Behaviors in Coding Agents.* arXiv:2609.30725.
7. J. Yang et al. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.* NeurIPS, arXiv:2405.15793.
8. C. S. Xia et al. (2024). *Agentless: Demystifying LLM-based Software Engineering Agents.* arXiv:2407.01489.
9. L. Chen et al. (2023). *Punica: Multi-Tenant LoRA Serving.* MLSys 2024, arXiv:2310.18547.
10. Y. Sheng et al. (2023). *S-LoRA: Serving Thousands of Concurrent LoRA Adapters.* MLSys 2024, arXiv:2311.03285.
11. W. Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, arXiv:2309.06180.
12. S.-Y. Liu et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML, arXiv:2402.09353.
13. E. Zelikman et al. (2022). *STaR: Bootstrapping Reasoning With Reasoning.* NeurIPS, arXiv:2203.14465.
14. A. Singh et al. (2023). *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models.* TMLR 2024, arXiv:2312.06585.
15. C. E. Jimenez et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, arXiv:2310.06770.
16. N. Jain et al. (2024). *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* ICLR 2025, arXiv:2403.07974.
17. R. Aleithan et al. (2024). *SWE-Bench+: Enhanced Coding Benchmark for LLMs.* arXiv:2410.06992.
18. T. Ahmed et al. (2025). *Investigating Test Overfitting on SWE-bench.* arXiv:2511.16858.
19. S. Robertson and H. Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval 3(4):333–389.
20. M. Günther et al. (2023). *Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents.* arXiv:2310.19923.
