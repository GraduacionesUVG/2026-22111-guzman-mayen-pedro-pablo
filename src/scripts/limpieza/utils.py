import re
import unicodedata
import pandas as pd
import numpy as np

def normalizar_texto(texto):
    """Normaliza cadenas de texto en español."""
    if not isinstance(texto, str):
        return texto
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn' or c == '\u0303')
    texto = texto.upper()
    texto = re.sub(r'[^A-Z0-9\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def describe_categorical_variables(dataframe):
    """Describe columnas categóricas del dataset"""
    for col in dataframe:
        print(f'Columna: {col}')
        print(f'-' * 50)
        print(f'Valores únicos {list(dataframe[col].unique())} | Frecuencia {dataframe[col].nunique()}')
        print(f'-' * 50)

def determinar_estado_curso(row):
    """Asigna APROBADO o REPROBADO según la nota final cuando el estado del curso está vacío"""
    nota = row['NotaFinal']
    estado = row['EstadoCurso']
    if pd.isna(estado):
        nuevo_estado = 'APROBADO' if nota >= 61 else 'REPROBADO'
        return nuevo_estado
    return estado

def determinar_genero(genero):
    """Imputa el género faltante con la categoría más frecuente (Masculino)"""
    if pd.isna(genero):
        return 'Masculino'
    return genero

def aprobado_a_binaria(texto):
    """Convierte la columna Aprobado de Sí/No a 1/0"""
    if texto == 'Sí':
        return 1
    return 0


def find_different_activities(array_to_search, reference_array):
    """Devuelve las actividades de array_to_search que no están en reference_array"""
    differents = []
    for i in array_to_search:
        if i not in reference_array:
            differents.append(i)

    return differents

def impute_new_registers(actvity_array, register_df, filtered_df):
    """Completa las actividades faltantes de cada estudiante repartiendo los puntos que le faltan para llegar a su nota final"""
    resultados = []
    lookup_actividad = filtered_df.groupby('Actividad').first()

    # Se procesa cada estudiante por separado
    for i in filtered_df['Est_ID'].unique():
        df_student = filtered_df[filtered_df['Est_ID'] == i].copy()
        # Actividades planificadas que el estudiante no tiene registradas
        actividades_faltantes = actvity_array[
            actvity_array['Actividad'].isin(
                find_different_activities(
                    actvity_array['Actividad'].values,
                    df_student['Actividad'].unique()
                )
            )
        ].copy()

        # Puntos que faltan para llegar a la nota final registrada
        nota_final = register_df.loc[register_df['Est_ID'] == i, 'NotaFinal'].iloc[0]
        nota_restante = nota_final - df_student['NotaPtos'].sum()
        puntos_posibles = actividades_faltantes['Puntos posibles']
        peso = puntos_posibles.sum()

        # Reparto proporcional a los puntos posibles de cada actividad
        if peso > 0:
            valor = nota_restante * puntos_posibles / peso
            # Ajuste: no menor a 0 ni mayor a los puntos posibles de la actividad
            valor = valor.clip(lower=0, upper=puntos_posibles)
        else:
            valor = pd.Series(0.0, index=actividades_faltantes.index)

        # Puntos que no se pudieron repartir por el recorte
        residuo = nota_restante - valor.sum()

        # Columnas de los nuevos registros
        actividades_faltantes['NotaPtos'] = valor
        actividades_faltantes['Nota'] = valor
        actividades_faltantes = actividades_faltantes.rename(columns={
            'Puntos posibles': 'PuntosPosibles',
        })
        actividades_faltantes['Est_ID'] = i
        actividades_faltantes['Anio'] = df_student['Anio'].unique()[0]
        actividades_faltantes['Seccion'] = df_student['Seccion'].unique()[0]
        actividades_faltantes['Ciclo'] = df_student['Ciclo'].unique()[0]
        actividades_faltantes['CantComentarios'] = 0
        actividades_faltantes['FechaEnvio'] = np.nan
        actividades_faltantes['NotaPorciento'] = np.where(
            actividades_faltantes['PuntosPosibles'] > 0,
            actividades_faltantes['NotaPtos'] / actividades_faltantes['PuntosPosibles'] * 100,
            0
        )
        actividades_faltantes['perteneceAzona'] = True
        actividades_faltantes['zona'] = True
        actividades_faltantes['imputado'] = True
        actividades_faltantes['residuo_estudiante'] = residuo

        # Columnas propias de la actividad, tomadas de otros registros
        for col in ['RevisionPares', 'TipoCalificacion', 'FechaTodoElDia', 'FechaVencimiento',
                    'TipoEntrega', 'FechaCalificacion', 'TipoEnvio']:
            actividades_faltantes[col] = actividades_faltantes['Actividad'].map(lookup_actividad[col])

        resultados.append(actividades_faltantes)

    return pd.concat(resultados)
