# Datos

> **Los datos no se incluyen en este repositorio.** Contienen registros académicos
> de estudiantes de la Universidad del Valle de Guatemala (notas, PAA, colegio de
> procedencia, género, título de nivel medio obtenido, etc). Por confidencialidad
> solo se publica el código. Para reproducir el proyecto, solicita los archivos al
> a Lynette García (lgarciap@uvg.edu.gt) y colócalos en esta carpeta con los nombres
> exactos que se indican abajo.

Todos los notebooks leen y escriben en esta carpeta mediante rutas relativas
(`../../Datos/...`), así que no hace falta modificar el código.

## Archivos de entrada (originales)

| Archivo | Contenido | Lo usa |
|---|---|---|
| `Datos_registro.RData` | Objeto `estudiantes_unicos`: un registro por estudiante (Est_ID, PAA, Colegio, TituloNivelMedio, NotaFinal, Aprobado, Genero, EstadoCurso, veces_curso). | `limpieza/Limpieza_registro.ipynb` |
| `pensamiento.RData` | Objeto `pensamiento_final`: una fila por actividad evaluada por estudiante, exportada de Canvas (2019–2022). | `limpieza/Limpieza_registro.ipynb`, `limpieza/Limpieza_notas.ipynb` |
| `Actividades Pensamiento.xlsx` | Una hoja por año (`2019`…`2022`) con la planificación oficial: nombre de cada actividad y sus puntos posibles. | `limpieza/Limpieza_notas.ipynb` |

## Archivos generados (en orden)

| Archivo | Generado por | Descripción |
|---|---|---|
| `registro_cleaned.csv` | `Limpieza_registro.ipynb` | Registro limpio: una fila por estudiante **y por intento** (los repitentes tienen una fila por año). |
| `actividades_cleaned.csv` | `Limpieza_notas.ipynb` | Actividades limpias, sin duplicados, con actividades faltantes imputadas y la bandera `entrega_tarde`. |
| `notas_pensamiento_cuantitativo.csv` | `feature_extraction.ipynb` | Dataset final de modelado: una fila por estudiante-intento con las características mensuales. |
| `entrenamiento.csv`, `validacion.csv`, `prueba.csv` | `analisis_exploratorio.ipynb` | División 70 / 15 / 15 estratificada por `Aprobado` (`random_state=42`). |
| `registro_prueba_herramienta.csv`, `notas_prueba_herramienta.csv` | `app/data/generacion_prueba.ipynb` | Archivos de entrada para probar la aplicación con los estudiantes del conjunto de prueba. |

## Diccionario de variables de los conjuntos originales

### Información de los estudiantes (`Datos_registro.RData`, objeto `estudiantes_unicos`)

Tamaño inicial: **2,720 filas y 9 columnas**. Cada fila representa a un estudiante.

| Variable | Descripción |
|---|---|
| `Est_ID` | Identificador único de cada estudiante (anonimizado) |
| `PAA` | Nota obtenida en el examen de admisión PAA |
| `Colegio` | Institución de nivel medio de la que proviene el estudiante |
| `TituloNivelMedio` | Variable categórica que describe el título de nivel medio obtenido por el estudiante |
| `NotaFinal` | Nota final obtenida por el estudiante al final del curso |
| `Aprobado` | Variable categórica que indica si el estudiante aprobó el curso (`Sí` / `No`); vacía cuando no hay información del curso |
| `Genero` | Género del estudiante |
| `EstadoCurso` | Indica si el estudiante aprobó, reprobó, se retiró o convalidó el curso al final del semestre (`APROBADO`, `REPROBADO`, `CURSO RETIRADO`, `CONVALIDADO`) |
| `veces_curso` | Número de veces que el estudiante ha tomado el curso (de 1 a 4) |

### Actividades realizadas por los estudiantes (`pensamiento.RData`, objeto `pensamiento_final`)

Tamaño inicial: **61,943 filas y 20 columnas**. Cada fila representa una actividad registrada para un estudiante en Canvas.

| Variable | Descripción |
|---|---|
| `Est_ID` | Identificador único de cada estudiante (anonimizado) |
| `Actividad` | Nombre de la actividad realizada |
| `Seccion` | Sección del curso a la que pertenece el estudiante |
| `Ciclo` | Indica si la actividad se realizó en el primer o segundo semestre (en estos datos solo hay primer semestre) |
| `zona` | Indica si la actividad pertenece a la zona final o no |
| `Anio` | Año en el que se realizó la actividad |
| `TipoCalificacion` | Indica si la nota de la actividad se expresa en porcentaje (`percent`) o en puntos netos (`points`) |
| `PuntosPosibles` | Cantidad máxima de puntos obtenibles en la actividad |
| `RevisionPares` | Indica si la actividad fue revisada por pares o no |
| `FechaTodoElDia` | Fecha de entrega de la actividad |
| `FechaVencimiento` | Fecha en la que la actividad se cierra por completo y deja de estar disponible para entrega por parte de los estudiantes |
| `TipoEntrega` | Indica si la actividad debía entregarse en línea, en papel o si era un *quiz* |
| `Nota` | Nota obtenida por el estudiante en la actividad |
| `FechaCalificacion` | Fecha en la que la actividad fue calificada |
| `FechaEnvio` | Fecha en la que el estudiante entregó la actividad |
| `TipoEnvio` | Indica si el estudiante envió la tarea subiendo un documento, de forma presencial o mediante un *quiz* |
| `CantComentarios` | Cantidad de comentarios registrados en la plataforma para la entrega de la actividad |
| `NotaPtos` | Nota obtenida por el estudiante en puntos netos |
| `NotaPorciento` | Nota obtenida en la actividad en porcentaje |
| `perteneceAzona` | Indica si la actividad pertenece a zona o no |
