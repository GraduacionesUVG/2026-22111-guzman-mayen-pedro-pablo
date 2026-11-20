import gradio as gr
from predict import predict
import tempfile

# Opciones del selector de mes
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo"]
MODALIDADES = ["Virtual", "Presencial"]

# Mensaje que se muestra antes de predecir
MENSAJE_INICIAL = "Sube tus datos para obtener predicciones"

# Estilo del mensaje inicial
CSS = """
.mensaje-inicial {
    text-align: center;
    padding: 56px 16px;
    font-size: 1.1rem;
    color: var(--body-text-color-subdued);
}
"""

# Texto de la pestaña de instrucciones
INSTRUCCIONES = """
## ¿Cómo usar la herramienta?

1. Elige el **mes** hasta el cual quieres evaluar el riesgo.
2. Sube el CSV de **información general** de los estudiantes.
3. Sube el CSV de **actividades** de los estudiantes.
4. Presiona **Predecir** y revisa los resultados.
5. Descarga los resultados con **Exportar resultados (CSV)**.

---

### 1. Mes
Lista desplegable con valores de **Enero** a **Mayo**. Es obligatorio.

- Solo se toman en cuenta las actividades del mes elegido y de los meses anteriores.
- Cada mes tiene su propio modelo, entrenado únicamente con la información disponible hasta ese mes. Para **Mayo** se usa el modelo de **Abril**.
- El archivo de actividades debe tener al menos una actividad en el mes elegido; si no, la herramienta pide elegir un mes anterior.

---

### 2. Información general de los estudiantes (`.csv`)
Un registro por estudiante. Codificación **UTF-8**, separador **coma**. Todas las columnas son obligatorias.

| Columna | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `Est_ID` | entero | Identificador único de cada estudiante (anonimizado) | `102345` |
| `Anio` | entero | Año en el que el estudiante lleva el curso | `2026` |
| `PAA` | numérico | Nota obtenida en el examen de admisión PAA (200–1600) | `1234` |
| `Colegio` | texto | Institución de nivel medio de la que proviene el estudiante, en **mayúsculas y sin tildes** | `COLEGIO METROPOLITANO` |
| `TituloNivelMedio` | texto | Título de nivel medio obtenido por el estudiante, en **mayúsculas y sin tildes** | `BACHILLER EN CIENCIAS Y LETRAS` |
| `veces_curso` | entero | Número de veces que el estudiante ha tomado el curso, incluida la actual | `1` |

> Si un colegio o título no aparece en los datos con los que se entrenó el modelo, se le asigna el valor promedio.
> El archivo puede traer otras columnas del registro (por ejemplo `Genero`); la herramienta solo usa las de esta tabla.

---

### 3. Actividades de los estudiantes (`.csv`)
Un registro por **actividad por estudiante**. Codificación **UTF-8**, separador **coma**.

| Columna | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `Est_ID` | entero | Identificador único del estudiante; debe coincidir con el archivo de información general | `102345` |
| `Anio` | entero | Año en el que se realizó la actividad, igual que en el archivo de información general | `2026` |
| `Seccion` | texto/entero | Sección del curso a la que pertenece el estudiante | `10` |
| `Actividad` | texto | Nombre de la actividad realizada, en **mayúsculas y sin tildes** | `TAREA 3` |
| `NotaPtos` | numérico | Nota obtenida por el estudiante en puntos netos | `1.5` |
| `PuntosPosibles` | numérico | Cantidad máxima de puntos obtenibles en la actividad | `2` |
| `FechaVencimiento` | fecha `YYYY-MM-DD` | Fecha en la que la actividad se cierra por completo y deja de estar disponible para entrega | `2026-02-16` |
| `FechaTodoElDia` | fecha `YYYY-MM-DD` | Fecha de entrega de la actividad (puede ir vacía) | `2026-02-14` |
| `FechaCalificacion` | fecha `YYYY-MM-DD` | Fecha en la que la actividad fue calificada (puede ir vacía) | `2026-02-18` |
| `FechaEnvio` | fecha `YYYY-MM-DD` | Fecha en la que el estudiante entregó la actividad (vacía si no entregó) | `2026-02-13` |

- El archivo puede traer las demás columnas del registro de actividades de Canvas (por ejemplo `TipoEntrega` o `CantComentarios`); la herramienta solo usa las de esta tabla.
- Las cuatro columnas de fecha deben existir aunque tengan celdas vacías. El mes de cada actividad se toma de `FechaVencimiento`; si está vacía, de `FechaTodoElDia` y, si no, de `FechaCalificacion`.
- Los exámenes se reconocen por su nombre y deben llamarse exactamente `PARCIAL 1`, `PARCIAL 2`, `PARCIAL 3`, `PARCIAL 4` o `EXAMEN FINAL`. Se asignan a febrero, marzo, abril y mayo (el parcial 4 y el examen final), respectivamente.

---

### Resultado
Una tabla con cada estudiante, ordenada de mayor a menor riesgo:

| Columna | Descripción |
|---|---|
| `Est_ID` | Identificador del estudiante |
| `Probabilidad de reprobar` | Probabilidad estimada por el modelo (0 a 1) |
| `Predicción` | **En riesgo** si la probabilidad supera el umbral del modelo de ese mes; si no, **Fuera de riesgo** |

> El umbral de cada mes se eligió para priorizar la detección de estudiantes en riesgo, aunque eso implique algunas falsas alarmas.
"""


async def predict_and_export(mes, info_file, actividades_file):
    """Ejecuta la predicción, muestra la tabla de resultados y guarda un CSV temporal para descargarlo"""
    df = await predict(mes, info_file, actividades_file)
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", prefix=f"riesgo_{mes}_"
    )
    df.to_csv(tmp.name, index=False, encoding="utf-8-sig")
    tmp.close()
    # Se oculta el mensaje inicial y se muestran los resultados
    return (
        gr.Markdown(visible=False),
        gr.Dataframe(value=df, visible=True),
        gr.DownloadButton(value=tmp.name, visible=True),
    )

# Interfaz de la aplicación
with gr.Blocks(title="Detección temprana de riesgo – Pensamiento Cuantitativo") as demo:
    gr.Markdown("# Detección temprana de riesgo académico")

    with gr.Tab("Predicción"):
        with gr.Row():
            mes = gr.Dropdown(choices=MESES, label="Elige un mes")
        with gr.Row():
            info_file = gr.File(label="Información general de los estudiantes", file_types=[".csv"])
            actividades_file = gr.File(label="Actividades de los estudiantes", file_types=[".csv"])
        btn = gr.Button("Predecir", variant="primary")
        mensaje = gr.Markdown(MENSAJE_INICIAL, elem_classes=["mensaje-inicial"])
        salida = gr.Dataframe(
            label="Resultados",
            wrap=False,
            max_height=500,
            interactive=False,
            visible=False,
        )

        exportar_btn = gr.DownloadButton(
            "Exportar resultados (CSV)",
            visible=False,
        )

        # Al presionar Predecir se ejecuta la predicción y se habilita la descarga
        btn.click(
            fn=predict_and_export,
            inputs=[mes, info_file, actividades_file],
            outputs=[mensaje, salida, exportar_btn],
        )

    with gr.Tab("Instrucciones"):
        gr.Markdown(INSTRUCCIONES)

# Inicio del servidor
demo.launch(css=CSS)
