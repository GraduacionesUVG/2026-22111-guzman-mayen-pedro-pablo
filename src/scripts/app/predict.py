import joblib
import os
import asyncio
import pandas as pd
import numpy as np
import gradio as gr

BASE_FEATURES = ['PAA', 'Colegio', 'TituloNivelMedio',  'veces_curso','DiferenciaPromedio', 'Virtualidad']

POS_LABEL = 1
NEG_LABEL = 0 
 
COLS_TARGET_ENCODING = ["TituloNivelMedio", "Colegio"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
# Modelos de cada mes con su encoder, umbral y columnas
BUNDLES = {
    i: joblib.load(os.path.join(BASE_DIR, "models", f"modelo_random_forest_mes{i}.pkl"))
    for i in range(1, 5)
}
diccionario_modelos = {
    1: 'Modelo mes 1',
    2: 'Modelo mes 2',
    3: 'Modelo mes 3',
    4: 'Modelo mes 4',
    5: 'Modelo mes 5'
}

# Número de cada mes
month_map = {
    'Enero': 1,
    'Febrero': 2,
    'Marzo': 3,
    'Abril': 4,
    'Mayo':5

}

async def convert_csv(archivo):
    """Lee un archivo CSV subido y lo convierte en DataFrame"""
    ruta = archivo if isinstance(archivo, str) else archivo.name
    return await asyncio.to_thread(pd.read_csv, ruta)

async def aplicar_encoders(X, te):
    """Aplica el target encoding del modelo a las columnas Colegio y TituloNivelMedio"""
    def transformar():
        """Copia los datos y transforma las columnas categóricas"""
        X_enc = X.copy().reset_index(drop=True)
        X_enc[COLS_TARGET_ENCODING] = te.transform(X_enc[COLS_TARGET_ENCODING])
        return X_enc
    return await asyncio.to_thread(transformar)

async def predict(month, registro, actividades):
    """Calcula la probabilidad de reprobar de cada estudiante con el modelo del mes elegido"""
    if not month or registro is None or actividades is None:
        raise gr.Error("Selecciona el mes y sube ambos archivos.")

    registro, actividades = await asyncio.gather(
        convert_csv(registro),
        convert_csv(actividades),
    )

    mes_num = month_map[month]
    # Para mayo se usa el modelo de abril
    mes_modelo = min(mes_num, 4)
    bundle = BUNDLES[mes_modelo]

    features = await preprocess_data(actividades, mes_num, registro)

    # Columnas en el orden que espera el modelo
    X = features.reindex(columns=bundle["columnas"])
    X = await aplicar_encoders(X, bundle["encoder"])

    modelo = bundle["modelo"]
    idx_reprobado = list(modelo.classes_).index(NEG_LABEL)
    probas = await asyncio.to_thread(modelo.predict_proba, X)
    p_reprobado = probas[:, idx_reprobado]
 
    # Clasificación con el umbral elegido en validación
    en_riesgo = p_reprobado >= bundle["umbral"]
 
    resultado = (
        pd.DataFrame({
            "Est_ID": features["Est_ID"].values,
            "Probabilidad de reprobar": np.round(p_reprobado, 3),
            "Predicción": np.where(en_riesgo, "En riesgo", "Fuera de riesgo"),
        })
        .sort_values("Probabilidad de reprobar", ascending=False)
        .reset_index(drop=True)
    )
    return resultado

def es_examen(texto):
    """Devuelve 1 si la actividad es un parcial o el examen final"""
    if texto in ['PARCIAL 1', 'PARCIAL 2', 'PARCIAL 3', 'PARCIAL 4', 'EXAMEN FINAL']:
        return 1
    return 0

def asignar_mes_a_examen(nombre_examen):
    """Devuelve el mes del semestre en que se realiza cada examen"""
    if nombre_examen == 'PARCIAL 1':
        return 2
    elif nombre_examen == 'PARCIAL 2':
        return 3
    elif nombre_examen == 'PARCIAL 3':
        return 4
    elif nombre_examen == 'PARCIAL 4' or nombre_examen == 'EXAMEN FINAL':
        return 5

def preprocess_data_helper(dataset, month, registro):
    """Calcula las características mensuales de cada estudiante con las actividades hasta el mes elegido"""
    # Marca de exámenes
    dataset['es_examen'] = dataset['Actividad'].apply(es_examen)
    # Conversión de fechas
    for col in ['FechaVencimiento', 'FechaTodoElDia', 'FechaCalificacion', 'FechaEnvio']:
        dataset[col] = pd.to_datetime(
            dataset[col],
            format='mixed',
            errors='coerce'
        )

    # Mes de cada actividad según sus fechas
    dataset['fecha_para_mes'] = (
        dataset['FechaVencimiento']
        .fillna(dataset['FechaTodoElDia'])
        .fillna(dataset['FechaCalificacion'])
    )

    dataset['mes'] = dataset['fecha_para_mes'].dt.month
    # Junio se agrupa con mayo
    dataset['mes'] = dataset['mes'].replace(6, 5)  

    # Actividades sin fecha se asignan al mes elegido
    sin_fecha = dataset['mes'].isna()
    dataset['mes'] = dataset['mes'].fillna(month)


    es_ex = dataset['es_examen'] == 1

    # Los exámenes toman el mes del calendario del curso
    mes_ex = dataset.loc[es_ex, 'Actividad'].apply(asignar_mes_a_examen)
    dataset.loc[es_ex, 'mes'] = mes_ex.fillna(dataset.loc[es_ex, 'mes'])
    dataset['mes'] = dataset['mes'].astype(int)

    # Debe existir al menos una actividad con fecha real en el mes elegido
    con_mes_real = ~sin_fecha | es_ex
    if not ((dataset['mes'] == month) & con_mes_real).any():
        nombre_mes = {v: k for k, v in month_map.items()}[month]
        raise gr.Error(
            f"El archivo de actividades no tiene datos de {nombre_mes}. "
            "Elige un mes anterior o sube un archivo más reciente."
        )

    dataset = dataset[dataset['mes'] <= month].copy() #Aplicamos filtro solo para actividades del mes seleccionado o anteriores.

    claves = ['Est_ID', 'Anio', 'Seccion']

    # Puntos de actividades sin contar exámenes
    dataset['ptos_act'] = dataset['NotaPtos'].where(~es_ex, 0)
    dataset['posibles_act'] = dataset['PuntosPosibles'].where(~es_ex, 0)

    # Agregación mensual por estudiante
    mensual = (
        dataset
        .groupby(claves + ['mes'], as_index=False)
        .agg(
            NotaPtos_mes=('NotaPtos', 'sum'),
            Nota_posible_mes=('PuntosPosibles', 'sum'),
            ptos_act_mes=('ptos_act', 'sum'),
            posibles_act_mes=('posibles_act', 'sum'),
        )
        .sort_values(claves + ['mes'])
    )

    g = mensual.groupby(claves)
    # Porcentaje acumulado en actividades
    mensual['porcentaje_acum_actividades'] = (
        g['ptos_act_mes'].cumsum() / g['posibles_act_mes'].cumsum().replace(0, np.nan) * 100
    )

    def a_ancho(df, valor, prefijo, ffill=True):
        """Pasa una variable mensual de formato largo a ancho, una columna por mes"""
        p = df.pivot_table(index=claves, columns='mes', values=valor)
        if ffill:
            p = p.ffill(axis=1) 
        p.columns = [f'{prefijo}_mes_{int(m)}' for m in p.columns]
        return p


    pivot_act = a_ancho(mensual, 'porcentaje_acum_actividades', 'porcentaje_acum_actividades')
    # Nota de cada examen en porcentaje
    ex = (
        dataset[es_ex]
        .groupby(claves + ['mes'], as_index=False)
        .agg(ptos=('NotaPtos', 'sum'), posibles=('PuntosPosibles', 'sum'))
    )
    ex['nota_examen'] = ex['ptos'] / ex['posibles'].replace(0, np.nan) * 100
    pivot_ex = a_ancho(ex, 'nota_examen', 'nota_examen', ffill=False)

    # Unión de las características mensuales
    features_mensuales = pivot_act.join([pivot_ex], how='outer')


    # Cambio mensual del porcentaje de actividades
    meses_act = sorted(int(c.rsplit('_', 1)[1]) for c in pivot_act.columns)
    for anterior, actual in zip(meses_act[:-1], meses_act[1:]):
        features_mensuales[f'delta_actividades_mes_{actual}'] = (
            features_mensuales[f'porcentaje_acum_actividades_mes_{actual}']
            - features_mensuales[f'porcentaje_acum_actividades_mes_{anterior}']
        )

    features_mensuales = features_mensuales.reset_index()

    assert not features_mensuales.duplicated(['Est_ID', 'Anio']).any(), \
        "Hay estudiantes en más de una sección el mismo año"

    # Unión con la información general
    registro = registro.merge(
        features_mensuales.drop(columns='Seccion'), on=['Est_ID', 'Anio'], how='left'
    )
    return registro

async def preprocess_data(dataset, month, registro):
    """Ejecuta el preprocesamiento en un hilo aparte para no bloquear la aplicación"""
    return await asyncio.to_thread(preprocess_data_helper, dataset, month, registro)

