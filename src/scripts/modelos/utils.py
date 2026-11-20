from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    fbeta_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error
)
import pickle
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, median_absolute_error, r2_score,
    recall_score, precision_score, f1_score, fbeta_score,
    balanced_accuracy_score, confusion_matrix, average_precision_score,
)

meses = {
    1: 'Enero',
    2: 'Febrero',
    3: 'Marzo',
    4: 'Abril',
    5: 'Mayo'
}

COLS_TARGET_ENCODING = ["TituloNivelMedio", "Colegio"]
POS_LABEL = 1   # Aprobado
NEG_LABEL = 0   # Reprobado (clase de interés)
NOMBRES_MODELOS = {
    "decision_tree": "Árbol de Decisión",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "logistic_regression": "Regresión Logística",
    "svm": "SVM",
    "sgd": "SGD",
    "linear_regression": "Regresión Lineal",
    "knn": "K-Vecinos Más Cercanos",
}

NOMBRES_CLASE = {NEG_LABEL: "Reprobado", POS_LABEL: "Aprobado"}


def obtener_importancias(model, columnas):
    """Obtiene la importancia de cada variable según el tipo de modelo y la normaliza para que sume 1"""
    columnas = list(columnas)
 
    if hasattr(model, "booster_"):                      
        imp = model.booster_.feature_importance(importance_type="gain")
        tipo = "gain"
    elif hasattr(model, "get_booster"):               
        scores = model.get_booster().get_score(importance_type="gain")
        imp = np.array([scores.get(c, 0.0) for c in columnas])
        tipo = "gain"
    elif hasattr(model, "feature_importances_"):        
        imp = model.feature_importances_
        tipo = "impurity"
    elif hasattr(model, "coef_"):                       
        imp = np.abs(model.coef_).ravel()
        tipo = "abs_coef"
    else:
        return None
 
    imp = np.asarray(imp, dtype=float)
    total = imp.sum()
    return pd.DataFrame({
        "variable": columnas,
        "importancia": imp,
        "importancia_norm": imp / total if total > 0 else imp,
        "tipo_importancia": tipo,
    }).sort_values("importancia_norm", ascending=False)

def guardar_conjuntos(out_dir, conjuntos, nombre_etiqueta="etiqueta"):
    """Guarda en CSV los conjuntos de datos usados en el entrenamiento"""

    for nombre, (X_, y_) in conjuntos.items():
        df = X_.copy()
        df[nombre_etiqueta] = np.asarray(y_)
        df[f"{nombre_etiqueta}_texto"] = df[nombre_etiqueta].map(NOMBRES_CLASE)
        df.to_csv(f"{out_dir}/{nombre}.csv", index=True)

def resolve_fixed_kwargs(fixed_kwargs, y_train):
    """Devuelve los parámetros fijos del modelo, evaluándolos si dependen de y_train"""
    if callable(fixed_kwargs):
        return fixed_kwargs(y_train)
    return dict(fixed_kwargs)


def nombre_legible(model_key):
    """Devuelve el nombre en español del modelo"""
    return NOMBRES_MODELOS.get(model_key, model_key)


def dividir_conjunto(X, y, estratificar=True, random_state=42):
    """Divide los datos en entrenamiento, validación y prueba (70/15/15) con estratificación opcional"""
    # Estratos: y (clasificación), un vector externo (regresión) o ninguno
    if isinstance(estratificar, bool):
        estratos = y if estratificar else None
    else:
        estratos = np.asarray(estratificar)

    if estratos is None:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=random_state
        )
    else:
        X_train, X_temp, y_train, y_temp, _, est_temp = train_test_split(
            X, y, estratos, test_size=0.30, random_state=random_state, stratify=estratos
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=est_temp
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def calcular_metricas_clasificacion(y_true, y_pred, y_proba_pos, y_proba_neg):
    """Calcula las métricas de clasificación de ambas clases, con énfasis en la clase reprobado"""
    cm = confusion_matrix(y_true, y_pred, labels=[NEG_LABEL, POS_LABEL])
    tn, fp, fn, tp = cm.ravel()


    # Recall y precisión de la clase reprobado a partir de la matriz de confusión
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    pr_auc = average_precision_score(y_true, y_proba_neg, pos_label=NEG_LABEL)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_proba_pos),
        "pr_auc": pr_auc,
        "precision_aprobado": precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "recall_aprobado": recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "f1_aprobado": f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "precision_reprobado": precision_neg,
        "recall_reprobado": recall_neg,
        "f1_reprobado": f1_score(y_true, y_pred, pos_label=NEG_LABEL, zero_division=0),
        "f2_reprobado": fbeta_score(y_true, y_pred, beta=2, pos_label=NEG_LABEL, zero_division=0),
        "matriz_confusion": cm,
    }

def calcular_metricas_regresion(y_true, y_pred, pesos):
    """Calcula MAE, MSE, RMSE y correlación ponderados por clase"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(pesos, dtype=float)
    m = {}

    # MAE y MSE ponderados
    m["mae"] = mean_absolute_error(y_true, y_pred, sample_weight=w)
    mse = mean_squared_error(y_true, y_pred, sample_weight=w)
    m["mse"] = float(mse)

    # RMSE ponderado
    m["rmse"] = float(np.sqrt(mse))

    # Coeficiente de correlación ponderado
    yt_c = y_true - np.average(y_true, weights=w)
    yp_c = y_pred - np.average(y_pred, weights=w)
    denom = np.sqrt(np.sum(w * yt_c ** 2) * np.sum(w * yp_c ** 2))
    m["corr"] = float(np.sum(w * yt_c * yp_c) / denom) if denom > 0 else float("nan")

    return m

def get_feature_importance(model, columnas):
    """Devuelve la importancia de las variables ordenada de mayor a menor"""
    if isinstance(model, Pipeline):
        model = model[-1]
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=columnas).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        coefs = np.atleast_2d(model.coef_)[0]
        return pd.Series(np.abs(coefs), index=columnas).sort_values(ascending=False)
    return None 


def graficar_matriz_confusion(cm, model_key, mes, out_dir):
    """Grafica y guarda la matriz de confusión"""
    etiquetas = ["Reprobado", "Aprobado"]
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Matriz de confusión — {nombre_legible(model_key)} ({meses[mes]})")
    plt.colorbar()
    plt.xticks([0, 1], etiquetas)
    plt.yticks([0, 1], etiquetas)
    for r in range(2):
        for c in range(2):
            plt.text(c, r, cm[r, c], ha="center", va="center", color="black")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/matriz_confusion_{model_key}_mes{mes}.png", bbox_inches="tight")
    plt.close()


def graficar_pr(y_true, y_proba_neg, pr_auc, model_key, mes, out_dir,
                umbral=None, punto_umbral=None, etiqueta_umbral="umbral"):
    """Grafica y guarda la curva precisión-recall de la clase reprobado marcando el umbral elegido"""
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba_neg, pos_label=NEG_LABEL)
    prevalencia = float(np.mean(np.asarray(y_true) == NEG_LABEL))
    plt.figure()
    plt.plot(recall_vals, precision_vals, drawstyle="steps-post", label=f"PR-AUC = {pr_auc:.3f}")
    if punto_umbral is not None:
        plt.scatter(*punto_umbral, color="red", zorder=5,
                    label=f"Punto de operación ({etiqueta_umbral} = {umbral:.3f})")
    plt.xlabel("Recall (sensibilidad)")
    plt.ylabel("Precisión")
    plt.title(f"Curva Precisión-Recall — {nombre_legible(model_key)} ({meses[mes]})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/curva_pr_{model_key}_mes{mes}.png", bbox_inches="tight")
    plt.close()


def graficar_importancia(model, columnas, model_key, mes, out_dir, sufijo=""):
    """Grafica y guarda las 15 variables más importantes del modelo"""
    importancias = get_feature_importance(model, columnas)
    if importancias is None:
        print(f"({model_key} no expone importancia de variables)")
        return
    plt.figure(figsize=(8, 6))
    importancias.head(15).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(f"Importancia de variables{sufijo} — {nombre_legible(model_key)} ({meses[mes]})")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/importancia_variables_{model_key}_mes{mes}.png", bbox_inches="tight")
    plt.close()


def aplicar_encoders(X, te, scaler=None):
    """Aplica el target encoding y el escalado ya ajustados a un conjunto nuevo"""
    X = X.copy().reset_index(drop=True)
    X[COLS_TARGET_ENCODING] = te.transform(X[COLS_TARGET_ENCODING])
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    return X


def guardar_modelo(model, model_key, mes, out_dir, te=None, scaler=None,
                   umbral=None, columnas=None):
    """Guarda el modelo junto con su encoder, scaler, umbral y columnas en un archivo pkl"""
    ruta = f"{out_dir}/modelo_{model_key}_mes{mes}.pkl"
    paquete = {
        "modelo": model,
        "encoder": te,
        "scaler": scaler,
        "umbral": umbral,
        "columnas": list(columnas) if columnas is not None else None,
        "model_key": model_key,
        "mes": mes,
    }
    with open(ruta, "wb") as f:
        pickle.dump(paquete, f)
    return ruta


def umbral_recall_minimo(y_true, proba_neg, recall_meta=0.80, pos_label=NEG_LABEL):
    """Busca el umbral con mayor precisión que alcance un recall mínimo"""
    prec, rec, thr = precision_recall_curve(y_true, proba_neg, pos_label=pos_label)
    prec, rec = prec[:-1], rec[:-1]         
    validos = rec >= recall_meta
    if not validos.any():
        idx = np.argmax(rec)
    else:
        idx = np.where(validos)[0][np.argmax(prec[validos])]
    f1 = 2 * prec[idx] * rec[idx] / (prec[idx] + rec[idx] + 1e-12)
    return thr[idx], f1, prec[idx], rec[idx]

def umbral_max_fbeta(y_true, proba_neg, beta=2.0, pos_label=NEG_LABEL):
    """Busca el umbral de decisión que maximiza F-beta para la clase reprobado"""
    prec, rec, thr = precision_recall_curve(y_true, proba_neg, pos_label=pos_label)
    prec, rec = prec[:-1], rec[:-1]           
    b2 = beta ** 2
    fbeta = (1 + b2) * prec * rec / (b2 * prec + rec + 1e-12)
    idx = np.argmax(fbeta)
    return thr[idx], fbeta[idx], prec[idx], rec[idx]

def graficar_pred_vs_real(y_true, y_pred, model_key, mes, out_dir):
    """Grafica y guarda la predicción contra el valor real"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="gray", label="Predicción perfecta")
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Predicción")
    ax.set_title(f"{nombre_legible(model_key)} – mes {mes}: predicción vs real")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/pred_vs_real.png", dpi=150)
    plt.close(fig)



def calcular_pesos(y, umbral=61):
    """Calcula pesos balanceados para que reprobados y aprobados aporten lo mismo al error"""
    y = np.asarray(y)
    reprobado = y < umbral
    n, n_rep = len(y), reprobado.sum()
    n_apr = n - n_rep
    return np.where(reprobado, n / (2 * n_rep), n / (2 * n_apr))


def graficar_residuos(y_true, y_pred, model_key, mes, out_dir):
    """Grafica y guarda los residuos contra la predicción y su distribución"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuos = y_true - y_pred
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, residuos, alpha=0.4, s=15)
    axes[0].axhline(0, color="gray", ls="--")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Residuo (real − predicción)")
    axes[0].set_title("Residuos vs predicción")
 
    axes[1].hist(residuos, bins=30, edgecolor="black")
    axes[1].set_xlabel("Residuo")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de residuos")
 
    fig.suptitle(f"{nombre_legible(model_key)} – mes {mes}")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/residuos.png", dpi=150)
    plt.close(fig)
 
 
def estimador_final(model):
    """Devuelve el último paso si el modelo es un Pipeline o el modelo si no lo es"""
    return model[-1] if hasattr(model, "steps") else model
