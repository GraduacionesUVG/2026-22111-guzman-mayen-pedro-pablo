"""Genera datos ficticios con el formato de entrada de la aplicación.

Los datos son aleatorios y no corresponden a ningún estudiante real. Sirven para
probar la herramienta sin los datos originales, que son confidenciales. Las
predicciones que se obtengan con ellos no tienen ningún significado.

Uso:
    python generar_datos_sinteticos.py
    python generar_datos_sinteticos.py --estudiantes 100 --semilla 7
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANIO = 2026
SECCIONES = [10, 20, 30]
COLEGIOS = ['OTRO', 'COLEGIO EJEMPLO A', 'COLEGIO EJEMPLO B', 'INSTITUTO EJEMPLO C']
TITULOS = ['BACHILLER EN CIENCIAS Y LETRAS', 'BACHILLER EN CIENCIAS Y LETRAS CON ORIENTACION EN COMPUTACION', 'OTRO']

# Planificación ficticia del semestre: cuatro actividades de 2 puntos por mes
# y un examen de 15 puntos por mes de febrero a mayo (suma 100 puntos)
DIAS_ACTIVIDAD = [5, 12, 19, 26]
EXAMENES = {2: 'PARCIAL 1', 3: 'PARCIAL 2', 4: 'PARCIAL 3', 5: 'PARCIAL 4'}


def planificacion():
    """Devuelve la lista de actividades del semestre con su mes, fecha de entrega y puntos posibles"""
    actividades = []
    numero = 1
    for mes in range(1, 6):
        for dia in DIAS_ACTIVIDAD:
            tipo = 'TAREA' if numero % 2 else 'EJERCITACION'
            actividades.append({
                'Actividad': f'{tipo} {(numero + 1) // 2}',
                'mes': mes,
                'entrega': pd.Timestamp(ANIO, mes, dia),
                'PuntosPosibles': 2.0,
                'es_examen': False,
            })
            numero += 1
        if mes in EXAMENES:
            actividades.append({
                'Actividad': EXAMENES[mes],
                'mes': mes,
                'entrega': pd.Timestamp(ANIO, mes, 25),
                'PuntosPosibles': 15.0,
                'es_examen': True,
            })
    return actividades


def generar_registro(n, rng):
    """Genera la información general de n estudiantes ficticios y su habilidad latente"""
    veces_curso = rng.choice([1, 2, 3], size=n, p=[0.88, 0.10, 0.02])
    # La habilidad latente define qué tan bien le va a cada estudiante;
    # los repitentes tienden a tener una habilidad menor
    habilidad = rng.normal(0, 1, size=n) - 0.8 * (veces_curso - 1)
    paa = np.clip(np.round(1240 + 110 * habilidad + rng.normal(0, 60, size=n)), 700, 1600)
    registro = pd.DataFrame({
        'Est_ID': np.arange(900001, 900001 + n),
        'Anio': ANIO,
        'PAA': paa,
        'Colegio': rng.choice(COLEGIOS, size=n),
        'TituloNivelMedio': rng.choice(TITULOS, size=n, p=[0.7, 0.2, 0.1]),
        'veces_curso': veces_curso,
    })
    return registro, habilidad


def generar_actividades(registro, habilidad, rng):
    """Genera una fila por actividad por estudiante con notas y fechas coherentes con su habilidad"""
    filas = []
    plan = planificacion()
    for est_id, h in zip(registro['Est_ID'], habilidad):
        seccion = rng.choice(SECCIONES)
        prob_entrega = np.clip(0.93 + 0.05 * h, 0.4, 0.99)
        for act in plan:
            entrega = act['entrega']
            vencimiento = entrega + pd.Timedelta(days=2)
            calificacion = entrega + pd.Timedelta(days=int(rng.integers(3, 10)))
            if act['es_examen']:
                # Exámenes: nota según la habilidad, sin fecha de envío
                porcentaje = np.clip(0.72 + 0.17 * h + rng.normal(0, 0.10), 0, 1)
                envio = pd.NaT
            elif rng.random() < prob_entrega:
                # Actividad entregada, a veces después de la fecha de entrega
                porcentaje = np.clip(0.86 + 0.08 * h + rng.normal(0, 0.10), 0, 1)
                retraso = int(rng.integers(1, 3)) if rng.random() < 0.08 else -int(rng.integers(0, 3))
                envio = entrega + pd.Timedelta(days=retraso)
            else:
                # Actividad no entregada
                porcentaje = 0.0
                envio = pd.NaT
            filas.append({
                'Est_ID': est_id,
                'Anio': ANIO,
                'Seccion': seccion,
                'Actividad': act['Actividad'],
                'NotaPtos': round(float(porcentaje) * act['PuntosPosibles'], 2),
                'PuntosPosibles': act['PuntosPosibles'],
                'FechaVencimiento': vencimiento.date(),
                'FechaTodoElDia': entrega.date(),
                'FechaCalificacion': calificacion.date(),
                'FechaEnvio': envio.date() if pd.notna(envio) else None,
            })
    return pd.DataFrame(filas)


def main():
    """Lee los argumentos, genera los datos y guarda los dos archivos CSV"""
    parser = argparse.ArgumentParser(description='Genera datos ficticios para probar la aplicación')
    parser.add_argument('--estudiantes', type=int, default=60, help='Cantidad de estudiantes ficticios')
    parser.add_argument('--semilla', type=int, default=42, help='Semilla para que los datos sean reproducibles')
    parser.add_argument('--salida', type=Path, default=Path(__file__).parent / 'ejemplo',
                        help='Carpeta donde se guardan los archivos')
    args = parser.parse_args()

    rng = np.random.default_rng(args.semilla)
    registro, habilidad = generar_registro(args.estudiantes, rng)
    actividades = generar_actividades(registro, habilidad, rng)

    args.salida.mkdir(parents=True, exist_ok=True)
    registro.to_csv(args.salida / 'registro_sintetico.csv', index=False, encoding='utf-8')
    actividades.to_csv(args.salida / 'actividades_sinteticas.csv', index=False, encoding='utf-8')
    print(f'{len(registro)} estudiantes y {len(actividades)} actividades ficticias guardados en {args.salida}')


if __name__ == '__main__':
    main()
