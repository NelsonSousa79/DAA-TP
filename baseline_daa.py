"""Elemento 1 - DAA: validação comum (5-fold estratificado), OOF e baselines.
Uso: python baseline_daa.py  (dados em ./data/)"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

SEED, N_FOLDS, TARGET = 42, 5, "AGE_RANGE"
DROP = ["PID", "HHID"]            # identificadores (únicos por linha)

train = pd.read_csv("data/dataset_train.csv")
test = pd.read_csv("data/dataset_test_X.csv")
y = train[TARGET].astype(int)
X = train.drop(columns=[TARGET] + DROP)
# colunas constantes (PROXY, COUNTRY, ...) não trazem informação
const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
X = X.drop(columns=const)
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]

def preprocessing():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="NA")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), cat),
    ])

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(X, y))     # os MESMOS folds para todo o grupo

def avaliar(modelo, X, y, nome):
    oof = np.zeros((len(y), 16)); fold_id = np.zeros(len(y), int); sc = []
    for k, (tr, va) in enumerate(FOLDS):
        m = clone(modelo).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])
        oof[np.ix_(va, m.classes_)] = p
        fold_id[va] = k
        sc.append(f1_score(y.iloc[va], m.classes_[p.argmax(1)], average="macro"))
    print(f"{nome}: Macro F1 = {np.mean(sc):.4f} ± {np.std(sc):.4f}  folds={np.round(sc,4)}")
    out = pd.DataFrame(oof, columns=[f"p{i}" for i in range(16)])
    out.insert(0, "id", train["PID"].values); out.insert(1, "y_true", y.values)
    out.insert(2, "fold", fold_id); out["exp"] = nome
    out.to_csv(f"oof_{nome}.csv", index=False)
    return sc

avaliar(DummyClassifier(strategy="prior"), X, y, "dummy_v1")
lr = Pipeline([("pre", preprocessing()),
               ("clf", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))])
avaliar(lr, X, y, "lr_v1")

# submissão de exemplo (treino total) -> CSV com RowId, AGE_RANGE
lr.fit(X, y)
Xt = test.drop(columns=DROP + const)
pd.DataFrame({"RowId": np.arange(len(test)), "AGE_RANGE": lr.predict(Xt)}).to_csv("submission_lr_v1.csv", index=False)
