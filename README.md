# DMC-Challenge-II

### Planteamiento del Problema

Queremos conocer en que grado los alumnos de una universidad recomendarían a su universidad.

Una universidad local desea conocer en que grado sus alumnos recomendarían su universidad como centro de estudios profesional. Por eso realizó una encuesta a una muestra de sus alumnos para conocer su opinión acerca de la universidad. Finalmente existe una pregunta la cual es: Recomendarías la universidad a tus amigos/conocidos? donde la respuesta va en una escala de 0 a 10 siendo 0: nunca los recomendaría y 10: si los recomendaría.

La escala se ha categorizado en:

[0-3] = No recomendaría (1)

[4-5] = Es poco probable que recomiende (2)

[6-8] = Muy probable recomendaría (3)

[9-10] = Sí recomendaría (4)

Se busca predecir que tanto recomendaría el alumno su universidad, siguiendo la escala antes mencionada.

Para esto tendrás que poner en práctica tus conocimientos haciendo modelos de predicción y minería de texto. Si consideras que es un problema complicado, no dudes en ponerte en contacto con http://www.dataminingperu.com/ con gusto te informaremos de los cursos de capacitación que ofrecemos para que puedas resolver estos problemas.


Link: https://inclass.kaggle.com/c/satisfaccion-en-la-universidad

### Descripción de los archivos

train_universidad.csv - the training set
test_universidad.csv - the test set
Data fields

COD_ENCUESTADO - Id del estudiante
- Nombre Campus - Campus donde se realizó la encuesta
- Nivel Actual - Tipo de estudios, presencial, PEA, etc
- Clave Carrera - Código de la carrera
- Ciclo - Ciclo en el que se encuentra
- Comentario - Comentario
- IND_GEA - Indicador si pertenece a grupo de excelencia académica (GEA)
- IND_DELEGADO - Indicador si es delegado o no
- CANT_CURSOS_MATRICU_SIN_INGLES - Cantidad de cursos en que se encuentra matriculado sin incluir inglés
- UOD_depostista_ind_deportista: Indicador si es deportista calificado
- NPS: Variable a predecir

Para la predicción se pueden usar todos los campos a excepción de COD_ENCUESTADO.

### Evaluation

La métrica a usar será LogLoss. 
