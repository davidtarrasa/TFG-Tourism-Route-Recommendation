# Evaluación offline (pendiente)

Incluir scripts para:
- Split train/test por usuario (dejar últimas visitas para test).
- Métricas top-N: HitRate@k, Recall@k, MRR/NDCG.
- Comparar modos: hybrid, content, item, markov, embed (y futuros ALS/BPR).
- Métricas espaciales: longitud media de ruta, dispersión, cumplimiento de filtros (precio/free).

TODO:
- Implementar `evaluate.py` con CLI (entrada: ciudad, k, tamaño de test).
- Registrar resultados y curvas por modo.
