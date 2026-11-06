# EXEMPLO de um modelo "recebido" de outro elemento (ex.: E2): Random Forest com encoding ordinal
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from common import prep_ordinal
def build(num, cat):
    return Pipeline([("pre", prep_ordinal(num, cat)),
                     ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                    n_jobs=-1, random_state=42))])
