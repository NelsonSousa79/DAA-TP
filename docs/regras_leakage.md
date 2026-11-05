# Regras anti-leakage 

•	**Regra 1**. Qualquer passo que aprende com os dados (imputação, escala, encoding, seleção de features, reamostragem, PCA…) ajusta-se apenas no treino de cada fold. Usar sempre Pipeline.
•	**Regra 2**. Nunca usar o dataset de teste do Kaggle para ajustar nada (nem médias, nem categorias).
•	**Regra 3**. Não usar identificadores (PID, HHID) como features.
•	**Regra 4**. Usar os mesmos folds (FOLDS partilhado) sempre que se comparem modelos.
•	**Regra 5**. Reamostragem (SMOTE, oversampling…) só no treino do fold; nunca no fold de validação.
•	**Regra 6**. Seleção de features e ajuste de hiperparâmetros (Optuna) dentro do treino; o fold de validação não pode influenciar a escolha de forma não controlada.
•	**Regra 7**. Não usar o leaderboard do Kaggle como substituto da validação local (há só 3 submissões/dia e o score público pode enganar).
•	**Regra 8**. Features construídas com estatísticas do alvo (por exemplo, média da idade por categoria) só podem usar o treino do fold.
•	**Regra 9**. Cada autor responsável por cumprir as regras nos seus modelos; a revisão cruzada de código verifica isto.
