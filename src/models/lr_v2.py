from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from common import prep_onehot
def build(num, cat):
    return Pipeline([("pre", prep_onehot(num, cat)),
                     ("clf", LogisticRegression(C=0.03, max_iter=2000, class_weight="balanced"))])
