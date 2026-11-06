from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from common import prep_ordinal
def build(num, cat):
    return Pipeline([("pre", prep_ordinal(num, cat)),
                     ("clf", HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100,
                                                            class_weight="balanced", random_state=42))])
