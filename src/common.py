"""Infraestrutura comum (E1): dados, folds, preprocessing, avaliação, registo.
Todos os modelos usam isto -> comparações justas. Nada aqui é apagado ao testar modelos novos."""
import os, datetime, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

SEED, N_FOLDS, TARGET = 42, 5, "AGE_RANGE"
DROP = ["PID", "HHID"]                      # identificadores: nunca usar como features
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)

def load():
    train = pd.read_csv(P("data", "dataset_train.csv"))
    test = pd.read_csv(P("data", "dataset_test_X.csv"))
    y = train[TARGET].astype(int)
    ids = train["PID"].values
    X = train.drop(columns=[TARGET] + DROP)
    const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=const)
    Xtest = test.drop(columns=DROP + const)
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return X, y, Xtest, ids, num, cat

def get_folds(X, y):                        # SEMPRE os mesmos folds (seed 42)
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))

def prep_onehot(num, cat):                  # para LR, SVM, MLP...
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="NA")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), cat)])

def prep_ordinal(num, cat):                 # para árvores/boosting (sem escala)
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="NA")),
                          ("oe", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat)])

def avaliar(modelo, X, y, ids, nome, autor="?", notas="", sobrescrever=False):
    """Valida com os folds comuns, grava oof/oof_<nome>.csv e regista em experiments.csv."""
    f_oof = P("oof", f"oof_{nome}.csv")
    if os.path.exists(f_oof) and not sobrescrever:
        raise FileExistsError(f"{nome} já existe. Usa um nome novo (ex.: {nome}_b) para não perder o anterior.")
    oof = np.zeros((len(y), 16)); fold_id = np.zeros(len(y), int); sc = []
    for k, (tr, va) in enumerate(get_folds(X, y)):
        m = clone(modelo).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])
        oof[np.ix_(va, m.classes_)] = p; fold_id[va] = k
        sc.append(f1_score(y.iloc[va], m.classes_[p.argmax(1)], average="macro"))
        print(f"  fold {k}: {sc[-1]:.4f}")
    out = pd.DataFrame(oof, columns=[f"p{i}" for i in range(16)])
    out.insert(0, "id", ids); out.insert(1, "y_true", y.values); out.insert(2, "fold", fold_id); out["exp"] = nome
    out.to_csv(f_oof, index=False)
    f1c = f1_score(y, oof.argmax(1), average=None)           # F1 por classe (OOF global)
    print(f"{nome}: Macro F1 = {np.mean(sc):.4f} ± {np.std(sc):.4f}")
    print("  F1 por classe:", np.round(f1c, 2).tolist())
    reg = P("experiments.csv"); nova = not os.path.exists(reg)
    clf = modelo.steps[-1][1] if hasattr(modelo, "steps") else modelo
    with open(reg, "a", encoding="utf-8") as f:
        if nova: f.write("exp_id,data,autor,modelo,features,preprocessing,hiperparametros,seed,folds,macroF1_cv,desvio_cv,kaggle,notas\n")
        f.write(f"{nome},{datetime.date.today()},{autor},{type(clf).__name__},,,,{SEED},{N_FOLDS},{np.mean(sc):.4f},{np.std(sc):.4f},,{notas}\n")
    return np.mean(sc), np.std(sc)

def submeter(modelo, X, y, Xtest, nome):
    """Treina com TODO o treino e gera o CSV para o Kaggle em submissions/."""
    f = P("submissions", f"submission_{nome}.csv")
    m = clone(modelo).fit(X, y)
    pd.DataFrame({"RowId": np.arange(len(Xtest)), "AGE_RANGE": m.predict(Xtest)}).to_csv(f, index=False)
    print("Submissão gerada:", f)
