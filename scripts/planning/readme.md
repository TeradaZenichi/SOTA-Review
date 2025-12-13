## Pipeline de scripts para análise de planejamento

A ordem recomendada de execução dos scripts é:

1. **1-merge_raw_planning.py**
   - Une todos os dados brutos (CSV/BibTeX) de Scopus, IEEE, ScienceDirect, Web of Science, etc. em um único CSV padronizado: `data/planning/merged_raw_planning.csv`.

2. **2-dedup_best_filled.py**
   - Remove duplicatas (por DOI ou título+ano), mantendo o registro mais completo de cada grupo. Saída: `data/planning/merged_raw_planning_dedup.csv`.

3. **3-prioritize.py**
   - Filtra por ano (>=2015), calcula score de prioridade e bucketiza (`high`, `medium`, `low`). Saída: `data/planning/merged_prioritized.csv`.

4. **4-possible_duplicates.py** (opcional)
   - Gera um relatório de possíveis duplicatas remanescentes por título normalizado e ano. Saída: `results/planning/possible_duplicates.csv`.

5. **5-abstract_stats.py**
   - Taggeia cada artigo com problemas/métodos (por palavras-chave em título+abstract). Salva contagens e matrizes em `results/planning/`.

6. **6-topic_map.py**
   - Cria o mapa de tópicos 2D (UMAP+KMeans) e salva clusters, coordenadas e resumos em `results/planning/`.

7. **7-extra_maps.py**
   - Gera mapas complementares: t-SNE, evolução temporal dos clusters, heatmap problema×método.

8. **8-pub_trend.py**
   - Conta publicações por ano e gera gráficos de tendência em `results/planning/`.

---

- Todos os scripts usam como entrada padrão o arquivo mais "processado" disponível (`merged_prioritized.csv` > `merged_raw_planning_dedup.csv` > `merged_raw_planning.csv`).
- O fluxo recomendado é rodar do 1 ao 8, mas scripts 4, 5, 6, 7 e 8 podem ser executados independentemente após o passo 3.
- O README anterior e instruções específicas de cada script estão nos cabeçalhos dos próprios arquivos.
