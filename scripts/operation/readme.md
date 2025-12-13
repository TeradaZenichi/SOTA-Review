# Operation pipeline scripts

Esta pasta contém scripts análogos aos da pasta `scripts/planning`, mas focados na construção do corpus de **operação em tempo real** do EV com BESS.

## Ordem sugerida

1. **1-merge_raw_operation.py**
   - Concatena todos os CSVs brutos localizados em `data/operation/*` para gerar `data/operation/merged_raw_operation.csv`.
2. **2-dedup_best_filled.py**
   - Remove duplicatas por DOI ou título+ano, justificando a seleção do registro mais completo. Saída: `data/operation/merged_raw_operation_dedup.csv`.
3. **3-prioritize.py**
   - Calcula `priority_score` e bucketiza (`high	ox`, `medium`, `low`) enfatizando trabalhos recentes com palavras-chave de operação. Gera `data/operation/merged_prioritized_operation.csv`.
4. **4-possible_duplicates.py**
   - Relatório de possíveis duplicatas restantes por título+ano. Salva em `results/operation/possible_duplicates.csv`.
5. **5-abstract_stats.py**
   - Taggea problemas/metodologias (MPC, heurísticas, ML supervisionado, RL) e produz contagens, matrizes e tendências em `results/operation/`.
6. **6-topic_map.py**
   - Gera mapa UMAP+KMeans com descrições de clusters e salva `topic_map.csv`, `topic_map_cluster_summary.csv`, PNG/PDF.
7. **7-extra_maps.py**
   - Gera t-SNE, share por ano e heatmap problema×método baseados no `topic_map.csv`.
8. **8-pub_trend.py**
   - Conta publicações por ano e gera gráficos `year_counts.png`/`pdf`.

Todos os scripts usam a pasta `data/operation` como fonte de dados e gravam resultados em `results/operation`. Eles podem ser executados em sequência ou individualmente, desde que os arquivos de entrada existam (o passo 3 depende da deduplicação e assim por diante).
