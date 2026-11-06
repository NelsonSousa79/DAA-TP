# Ensemble simples: média das probabilidades de LR (one-hot) e HistGradientBoosting (ordinal).
# Pesos iguais, nada ajustado nos dados -> sem leakage.
from sklearn.ensemble import VotingClassifier
import lr_v2, hgb_v1
def build(num, cat):
    return VotingClassifier([("lr", lr_v2.build(num, cat)), ("hgb", hgb_v1.build(num, cat))],
                            voting="soft")
