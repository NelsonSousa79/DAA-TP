# DAA-TP — Previsão de faixa etária (Competição ML@DAA 2026/2027)

Modelos de Machine Learning que preveem a faixa etária (16 classes, AGE_RANGE 0–15)
a partir de um inquérito nacional de 2019. Métrica: Macro F1.

## Instalação
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt

## Dados
Descarregar o zip da competição no Kaggle e extrair para `data/`:
`dataset_train.csv`, `dataset_test_X.csv`, `example_submission_10.csv`.
Os dados não vão para o git.

## Correr o baseline (a partir da raiz do projeto)
    python src/baseline_daa.py
Gera `oof/oof_dummy_v1.csv`, `oof/oof_lr_v1.csv` e `submissions/submission_lr_v1.csv`.

## Protocolo comum
- StratifiedKFold, 5 folds, seed 42 (os mesmos folds para todos)
- Preprocessing sempre dentro de um Pipeline (ajustado só no treino de cada fold)
- OOF: id, y_true, fold, p0..p15, exp

## Estrutura
data/ (CSV, fora do git) · src/ (código) · notebooks/ (EDA) · oof/ (previsões OOF)
submissions/ (ficheiros Kaggle) · docs/ (regras de leakage) · experiments.csv (registo)

## Resultados
| Experiência | Macro F1 (CV) | Kaggle |
|---|---|---|
| dummy_v1 | 0.011 | — |
| lr_v1 | 0.430 ± 0.010 | 0.44 |
