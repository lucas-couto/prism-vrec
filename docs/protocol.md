# Experimental Protocol — Declarations

This document records every methodological decision the protocol fixes
explicitly, in the order a reviewer would ask about them. Each item is
implemented in code (pointers included) and must be restated in the
dissertation's methodology chapter.

## 1. Native dimensionality at extraction; learned projection `E` in the recommender

Comparing backbones is only valid if the backbone is the sole variable.
An earlier version of the framework forced every extractor through a
shared `Linear+ReLU` projection to a common dim — and in the frozen
condition that projection was **never trained** (a seeded random
projection), so the benchmark compared "backbone × random compression",
not backbones.

This protocol saves the **native** feature of each backbone at
extraction; the learned projection `E` inside each recommender (VBPR's `W_vis`,
DeepStyle's linear style projection, VNPR's visual transform, ACF's
component projection) maps `D_backbone → d`, trained jointly by the BPR loss with
the backbone frozen (fine-tuning end-to-end would be DVBPR, out of
scope). `d` (`common.visual_dim`) is fixed and identical across all
backbones of a comparison.

| Backbone | Weights (exact) | Extraction point | Native dim |
|---|---|---|---|
| ResNet-50 | torchvision `IMAGENET1K_V2` | global avg pool (after layer4) | 2048 |
| ConvNeXt-Base | timm `convnext_base.fb_in22k_ft_in1k` | global avg pool | 1024 |
| ViT-B/16 | timm `vit_base_patch16_224.augreg2_in21k_ft_in1k` | **CLS token** | 768 |
| CoAtNet-0 | timm `coatnet_0_rw_224.sw_in1k` | global avg pool | 768 |
| DINOv2 ViT-B/14 | torch.hub `facebookresearch/dinov2` (pinned commit) | **CLS token** | 768 |
| LeViT-256 | timm `levit_256.fb_dist_in1k` | pooled final-stage tokens | **512** (the "256" in the name is the stage-1 width) |
| CLIP ViT-B/32 | open_clip `laion2b_s34b_b79k` | **projected output (512, the `encode_image` space)** — the canonical practical use of CLIP as an extractor; the 768-d pre-projection width is NOT used | 512 |
| CvT-13 | HF `microsoft/cvt-13` (224px) | **CLS token** (the `[B, 384, 14, 14]` spatial map is used only as ACF components, never flattened into the pooled feature) | 384 |

Native dims are **read from the model** by a probe forward
(`BaseExtractor._probe_native_dim`), never hardcoded, and validated
against `configs/extractors.yaml` (`raw_dim`) — a mismatch fails the
extraction loudly. Every artifact ships a `.meta.json` sidecar
(backbone, native dim, extraction point, exact weights id, transform
recipe); the loader cross-checks features against it.

## 2. Canonical per-backbone preprocessing

The preprocessing recipe is part of the model. Three **distinct
normalisations** coexist across the 8 backbones:

- ImageNet (`0.485/0.456/0.406`): ResNet-50, ConvNeXt, LeViT, CvT, DINOv2
- Inception-style (`0.5/0.5/0.5`): **ViT-B/16 (augreg2)**, **CoAtNet-0 (sw_in1k)**
- CLIP (`0.48145466/…`): CLIP ViT-B/32

That earlier version applied ImageNet normalisation + direct bilinear
224 resize to all timm backbones — ViT-B/16 and CoAtNet-0 ran silently
degraded, and no backbone used its canonical bicubic resize+crop. This
protocol resolves each transform from the library that ships the weights (torchvision
`weights.transforms()`, `timm.data.resolve_model_data_config`,
`AutoImageProcessor`, open_clip's `preprocess`; DINOv2 is the one
hand-built recipe, matching its reference eval transform: resize 256
bicubic → crop 224, ImageNet norm). Pinned by
`tests/test_canonical_transforms.py`.

**Resolution posture (declared): all backbones consume 224×224 crops**,
their canonical eval resolution — the resize path (resize size,
interpolation, crop_pct) differs per recipe and is recorded in each
artifact's metadata. No hidden resolution variable.

## 3. Evaluation protocol: full ranking default, sampled opt-in

`full_ranking` is the default and the only protocol for reported
numbers (Krichene & Rendle, KDD 2020: sampled metrics can invert model
rankings). `sampled` exists for fast iteration only and is locked when
used: `n_negatives`, `negative_sampling_seed` (per-user seeded pools →
identical across models, required by the paired tests), sampling from
items unseen by the user. **Every recorded result row carries a
`protocol` column**; train-time BPR negative sampling is a different
thing entirely and is not configurable here.

**Model selection on validation.** Early stopping and the Optuna
objective (`ndcg@10`) score the **validation** held-outs, never the
test set: the training path loads `val.csv` and masks each user's train
items (`src/steps/train.py`, `src/utils/parallel.py`;
`src/utils/training.py` builds the selection `Evaluator`). The test set
is read only by the final evaluate step (`src/steps/evaluate.py`), so
hyperparameters and the stopping epoch are never chosen by looking at
test performance — the reported test numbers are an out-of-sample
estimate, not an optimistically-biased one. During validation the
user's own test item stays in the candidate set and competes as an
ordinary item; this is neutral across models and leaks nothing to the
model (the model never sees which items are held out).

**Training-time validation subsample (`common.eval_sample_size = 2000`).**
Selection scores a fixed subset of 2000 **validation** users instead of
all of them. The subset is drawn once per dataset, deterministically
(dedicated `np.random.default_rng`, `sample_seed` = global run seed —
not the per-trial job seed; `src/evaluation/protocol.py`), and is
identical for every model/embedding/trial, so selection remains a
paired comparison on a common validation-user set; only its variance
changes (standard error on ndcg@10 stays well below between-config
gaps). The validation metric is still full-ranking over all items for
those users. The final evaluate step constructs its `Evaluator` without
`sample_size` and ranks the entire test set.

## 4. Deterministic tie-breaking

All three ranking paths (batched torch, sampled numpy, single-user
numpy) implement one rule: exact-score ties are broken by a **fixed
random permutation of the item ids**, drawn once per run from a
dedicated `np.random.default_rng(seed)` (global run seed, not the
per-trial job seed) and shared by every model/trial of a `(dataset,
seed)` run. Item id is NOT used as the tie-break: `item_idx` correlates
with popularity in the DVBPR splits (Spearman −0.34 to −0.45), so an id
tie-break would systematically favour popular items inside a tie block —
penalising models with mass exact-ties (pure BPR over cold items) more
than models with distinct visual scores. In the sampled path ties are
likewise NOT broken by pool position (positives come first — that would
inflate metrics). When the held-out item is not tied, the returned rank
is identical to a plain descending sort. Each evaluation logs the
fraction of held-outs in an exact-score tie and the mean/max tie-block
size, so the real exact-tie frequency is measured during the battery.

## 5. Statistics

- **Wilcoxon signed-rank, `zero_method="pratt"`**: per-user LOO metrics
  are 0/1-heavy; the scipy default drops all zero differences,
  shrinking the effective sample far below `n_users`. Pratt keeps them.
  Every pairwise table reports `n_pairs` and `n_nonzero_pairs`.
- **Comparison families** (`src/evaluation/comparison_families.py`):
  the Holm correction and the Friedman omnibus are applied WITHIN the
  family of comparisons one research question defines — never over the
  Cartesian product of every config (all-pairs Holm over ~77 configs
  runs with `m ≈ 2900` and rejects everything artificially). Each
  family varies exactly one dimension: `backbone_within_model`
  (`m = C(n_backbones, 2)` per recommender), `model_within_backbone`,
  `fusion_within_model`, `frozen_vs_finetuned` (one `m = 1` pair per
  config). Every result row carries `family`, `group` and
  `n_comparisons_in_family` so the correction is auditable; `all_pairs`
  exists as an exploratory option only.
- **Primary metrics under LOO**: with one relevant item per user only
  two independent signals exist — hit-or-not (recall@k ≡ HitRate@k) and
  hit rank (ndcg@k). precision@k = recall@k / k and map@k = 1/rank are
  deterministic transforms; they stay in the raw evaluation CSVs, are
  excluded from the reported tests by default
  (`statistical.include_derived_metrics`), and must never be read as
  independent evidence.
- Friedman as the non-parametric omnibus (no normality assumption over
  per-user metric distributions), Holm–Bonferroni for multiple
  comparisons (uniformly more powerful than Bonferroni at the same
  FWER), percentile bootstrap CIs.
- **Effect size: Cliff's delta is primary** (non-parametric,
  tie-robust; thresholds 0.147/0.33/0.474) — consistent with
  Wilcoxon+pratt on zero-dominated differences. Cohen's d is parametric
  and inflates on such vectors (the std shrinks); it is off by default
  and available for diagnostics only.
- **Paired-difference bootstrap CI** on every pairwise row
  (`diff_mean`, `diff_ci_lower/upper`, resampling USERS): the CI that
  must agree with the Wilcoxon verdict. Per-config CIs are descriptive
  — under paired inference, overlapping individual CIs do NOT imply
  absence of a significant difference.

## 6. Fusion pipeline (Pipeline B — separate from the 8-extractor Pipeline A)

Sources: ResNet-50 (2048) + ViT-B/16 (768), native.

- **Element-wise family (8 of 11 strategies)** requires alignment, and
  the alignment method is an experimental variable
  (`alignment.method`): `learned` (default) — per-source
  `Linear(D_i→D)` co-trained via BPR (`LearnedAlignmentFusion`), the
  analogue of `E`; or `pca` — per-source PCA to `D`.
- **Concat family** operates on native dims: `concat` → 2816-d;
  `pca` (joint) reduces the 2816-d concat; **`pca_per_model`
  CONCATENATES after per-source PCA (→ `M·k`)** — declared, it is a
  concatenation-family strategy.
- **PCA protocol**: every PCA (`pca`, `pca_per_model`, `pca` alignment)
  is **fit exclusively on items with ≥1 training interaction** and
  applied to all items; seed fixed; cumulative explained variance
  logged per fit. The `k` of the PCA is itself a confounder vs the
  2816-d concat — report explained variance and/or sweep `k`.
- The fused `h_i` enters the recommender as the item's visual feature,
  through the same `E` as any single extractor.

## 7. Model-specific declarations

- **ACF is NOT degenerate**: it consumes `(n_items, M, D_native)`
  component artifacts (`*_comp.npy`; M = 49–256 depending on the
  backbone), so component-level attention has real components to
  attend. Its user-history side is built from train interactions only.
- **DeepStyle (paper-faithful)**: the item style term is
  `θ_i = E·f_i − c_cat(i)` — a linear projection `E` (`D_backbone → d`)
  minus a **learned category embedding** subtracted in the style space,
  as in the original paper. On the Amazon datasets, whose per-item
  category varies (declared `expects_categories: true`), this makes
  DeepStyle differ from VBPR. On Tradesy, which has no category
  (`expects_categories: false`, enforced at preprocess), every item maps
  to a single null category, so `c_cat(i)` is constant across items;
  the `α_u·c₀` term is item-independent and cancels in every BPR
  pairwise comparison, so DeepStyle **analytically degenerates into
  VBPR**. This is the expected, verified behaviour (see
  `tests/recommenders/test_deepstyle_paper.py::TestTradesyDegeneration`),
  not a bug. An earlier MLP-style variant (which did not subtract a
  category vector) was removed in commit `60c7436`.
- **Trainable-parameter counts differ across backbones** because `E`'s
  input is the native dim — an expected second-order effect, reported
  per cell (`n_trainable_params` column), never hidden.

## 8. Known confounder to acknowledge (defense question 13)

CLIP and DINOv2 are both ViT-B under the hood; if CLIP wins, the design
cannot separate architecture from pre-training data (2B image-text
pairs). The honest claim: this benchmark compares **extractors as
available in practice** (architecture + weights + canonical recipe),
not pure architectures.

## 9. Recommended robustness checks (before the defense)

Run the comparison at ≥2 values of `d` (64, 128), under both protocols,
and with multiple seeds (`seeds: [...]` is supported), verifying the
backbone ranking is stable. Each of these preempts a standard committee
question.

## 10. Declarações de protocolo (pt-BR — para a dissertação)

Declarações fechadas pelas auditorias de diagnóstico, em tom de
protocolo, prontas para migrar ao capítulo de metodologia.

### 10.1. Escopo do ajuste fino (fine-tuning) dos backbones

O ajuste fino dos backbones visuais é supervisionado exclusivamente pela
categoria dos itens e utiliza, de forma transdutiva, as imagens e os
rótulos de categoria de todo o catálogo — incluindo itens posteriormente
reservados para validação e teste da tarefa de recomendação —, sem em
nenhum momento acessar as interações usuário-item nem quais itens compõem
os conjuntos de validação/teste. Trata-se de uso a priori de metadados de
catálogo (equivalente ao emprego de representações pré-treinadas sobre
todos os itens), e não de vazamento de sinal de interação; a partição de
recomendação (treino/validação/teste do DVBPR) permanece fixa e é
consumida apenas nas etapas posteriores.

### 10.2. Seleção de modelo em validação

A seleção de hiperparâmetros e a parada antecipada (early stopping,
ndcg@10) são conduzidas sobre os usuários de **validação**, mascarando os
itens de treino de cada usuário. O conjunto de teste não é acessado em
nenhum momento do treinamento ou da seleção — é consumido apenas na
avaliação final —, de modo que as métricas reportadas são uma estimativa
fora da amostra, não enviesada pela seleção. Durante a validação, o item
de teste do usuário permanece no conjunto de candidatos e compete como um
item qualquer; isso é neutro entre os modelos e nada revela ao modelo.

### 10.3. Normalização pré-fusão

Antes de qualquer fusão element-wise, cada fonte é L2-normalizada por
vetor (`normalize_before_fusion: true`, padrão). Justificativa: a razão
entre as normas médias das fontes medida é de ~16× nas features brutas e
~9× após o alinhamento por PCA — sem a normalização, a fonte de maior
norma dominaria as fusões aditivas, tornando a comparação entre
estratégias um artefato de escala.

### 10.4. Desempate no ranking

Empates exatos de score são resolvidos por uma permutação aleatória fixa
dos itens, seedada a partir do seed global do run e compartilhada por
todos os modelos e trials de uma execução `(dataset, seed)`.
Justificativa: a correlação de Spearman entre `item_idx` e popularidade é
de −0,34 a −0,45 nos quatro datasets, o que tornaria o desempate por id
equivalente a um desempate por popularidade. A frequência real de empates
exatos é medida e registrada durante a própria bateria.

### 10.5. Conjunto de candidatos

A avaliação é full ranking sobre o catálogo inteiro, incluindo itens sem
qualquer interação de treino — itens frios com representação visual real
fazem parte do objeto de estudo do benchmark. Na validação, o item de
teste do usuário permanece como candidato (ver 10.2).

### 10.6. Orçamento e fluxo da busca de hiperparâmetros

O orçamento da busca de hiperparâmetros (número de trials, métrica de
seleção ndcg@10 em validação, paciência, épocas máximas, tamanho do
subsample de validação) é **uniforme para todos os recomendadores de um
mesmo dataset** — configurado numa fonte compartilhada única, nunca por
modelo; apenas os espaços de busca são por modelo, pois cada um tem seus
próprios hiperparâmetros. A busca completa é executada **apenas na seed
primária** de cada dataset; nas demais seeds, a melhor configuração
encontrada é re-treinada (replay), com parada antecipada em validação
ativa por seed. A avaliação final de cada célula consome o checkpoint do
melhor trial (cujo early stopping já rodou em validação) — não há
re-treino pós-busca, e o procedimento é idêntico para todos os modelos.
O conjunto de teste permanece intocado até a avaliação final.
