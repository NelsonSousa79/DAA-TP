"""Uso (a partir da raiz do projeto):
   python src/run.py rf_v1 --autor E2                # valida e regista
   python src/run.py rf_v1 --autor E2 --submit       # valida e gera o CSV para o Kaggle
O argumento é o nome do ficheiro em src/models/ (sem .py)."""
import sys, os, argparse, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
import common

ap = argparse.ArgumentParser()
ap.add_argument("modelo"); ap.add_argument("--autor", default="?"); ap.add_argument("--notas", default="")
ap.add_argument("--submit", action="store_true")
a = ap.parse_args()

X, y, Xtest, ids, num, cat = common.load()
modelo = importlib.import_module(a.modelo).build(num, cat)
common.avaliar(modelo, X, y, ids, a.modelo, autor=a.autor, notas=a.notas)
if a.submit:
    common.submeter(modelo, X, y, Xtest, a.modelo)
