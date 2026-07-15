# Runbook da bateria (Task I)

Guia operacional para rodar a bateria completa em instâncias
interrompíveis (spot), retomar após interrupção, acompanhar o progresso e
tratar falhas.

## Antes de lançar

1. **Extração pronta.** As features de cada `(dataset, backbone)` e as
   matrizes fundidas devem existir em `data/embeddings/<dataset>/`.
2. **Gate de sanidade das features** (Task G) — falha alto antes de
   queimar crédito:
   ```
   uv run python main.py --validate-features
   ```
   Sai com código ≠ 0 e mensagem clara se alguma matriz estiver corrompida
   (NaN/Inf, shape/dim/dtype errados, linha zerada). O `train`/`fuse`
   também validam automaticamente na entrada.
3. **Storage do Optuna persistente.** Já configurado em
   `configs/recommenders.yaml` (`storage: sqlite:///results/optuna/battery.db`)
   — os trials sobrevivem a um restart e a busca retoma de onde parou.

## Lançar

```
uv run python main.py --battery
```
O runner:
- **enumera** as células (datasets × configs visuais × recomendadores ×
  seeds) com as regras embutidas: BPR roda 1× por `(dataset, seed)`; AVBPR
  fora; DeepStyle roda no Tradesy; a **seed primária carrega a busca** e as
  demais são **replay** da melhor config (Task H);
- pula células já concluídas (idempotência: artefato per-user válido);
- registra o estado de cada célula no **manifest**
  `results/battery/manifest.json` (inspecionável).

No Docker, acompanhe com `docker logs -f prism-vrec` (não `docker compose
logs` — ver `docs/protocol.md` sobre a barra de progresso).

## Retomar após interrupção

Basta **relançar o mesmo comando**:
```
uv run python main.py --battery
```
Células `done` são puladas; o treino retoma do último checkpoint e a busca
do storage do Optuna. Nada concluído é refeito.

## Acompanhar progresso e projeção de custo

```
uv run python main.py --battery-status
```
Imprime a contagem por estado (`pending/running/done/failed`) e a
**estimativa de horas restantes** (duração média por tipo de célula ×
pendentes). Papéis sem amostra concluída ainda são reportados como “sem
estimativa”, nunca chutados.

## Falhas e retry

Uma célula que falha é isolada (as outras continuam) e marcada `failed` no
manifest com a mensagem de erro. Para reprocessar só as que falharam:
```
uv run python main.py --battery --retry-failed
```

## Onde ficam os artefatos e metadados

- **Per-user (F):** `results/per_user/<dataset>/<cell_key>.csv.gz` (rank do
  held-out, n_candidates, tie_block_size, top-20) + `<cell_key>.meta.json`
  (dataset, config visual, recomendador, seed, d, versão do protocolo).
- **Manifest:** `results/battery/manifest.json` (estado + `git_sha`,
  `git_dirty`, durações por célula).
- **Checkpoints do melhor trial:** `results/models/<dataset>/`.
- **Studies do Optuna:** `results/optuna/battery.db`.

Qualquer métrica de acurácia é **recomputável** a partir do rank
persistido, para qualquer `k`, sem GPU (`src/evaluation/derive_metrics.py`);
a matriz pareada usuários × sistemas para os testes estatísticos sai de
`src/evaluation/paired_loader.py`.
