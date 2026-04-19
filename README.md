# Clasificador de flores Kevin Josué Martínez Leyva A01067611

## Descripción del proyecto

El objetivo del clasificador de flores es mediante la utilización de machine learning poder clasificar fotografías de flores y determinar si la fotografía es alguna de las siguientes flores “Black eyed Susan, Calendula, California poppy, Common Daisy, Coreopsis, Daffodil, Dandelion, Iris, Magnolia, Rose, Sunflower, Tulip, Water Lily”. 

Se busca imitar funciones como las que poseen aplicaciones como “google photos”.

## Descripción del “dataset”

El “dataset” utilizado proviene de [Flower Classifiaction](https://www.kaggle.com/datasets/supriyoain/flower-classification)  el cual contiene 17229 imágenes, todas poseen un tamaño de 256x256, se ecuentran clasificadas en 19 clases. No todas las clases poseen la misma cantidad de imágenes, la distribución original del dataset se puede observar en la siguiente gráfica y tabla.

<img width="715" height="444" alt="Captura de pantalla 2026-04-12 213927" src="https://github.com/user-attachments/assets/1d80fbd0-c02f-45fa-862a-9abb2adebc07" />

<img width="461" height="460" alt="Captura de pantalla 2026-04-12 220131" src="https://github.com/user-attachments/assets/3c366fc5-dabd-42b8-a131-1522f52deb67" />

Para generar un mejor balance entre clases se eliminó aquellas clases que su porcentaje de representación sea menor a 5.4%, es un porcentaje alto, pero este porcentaje logró generar una distribución en donde las diferencia entre el porcentaje de representación ideal y el real posee una diferencia máxima de 0.35%. En la siguiente gráfica y tabla se puede observar la nueva distribución del dataset.

<img width="462" height="334" alt="Captura de pantalla 2026-04-12 221007" src="https://github.com/user-attachments/assets/ecc94b00-4213-466a-a547-4f709df79290" />
<img width="715" height="445" alt="Captura de pantalla 2026-04-12 221014" src="https://github.com/user-attachments/assets/c83a0dde-4aac-4c8c-855e-2a99d5aacac4" />


La pérdida de las clases eliminadas no es relevante para el objetivo del modelo. El “dataset” actual posee 13 clases y 13207 datos.

## División de datos

Con el “dataset” balanceado se realizo la separación de los datos de “train” y “test”, siguiendo un ratio de 80-20, para lograr esto se utilizo el script de random_test_train.py.

## Data Augmentation

Para incrementar la cantidad de datos (imágenes) con las que se entrenará al modelo se utilizaron técnicas de “data augmentation” que describiré a continuación:

- “Rescaling”: Todas las imágenes fueron normalizadas a un rango de 0 a 1, esto se logró dividiendo cada píxel entre 255.
- “Resizing”: Las imágenes se convirtieron a un tamaño de 120x120 pixeles, para mejorar la velocidad de entrenamiento del modelo.
- “Rotation”: Las imágenes fueron rotadas de manera aleatoria entre 10 y -10 grados.
- “Width shift”: Las imágenes se desplazaron un 20% de su ancho de manera horizontal.
- “Zoom”: Un zoom de entre 0.7 a 1. 3 del tamaño original.
- “Horizontal flip”: Se invirtió de manera horizontal a las imágenes, de forma aleatoria.

La razón de selección de estos filtros se debe a que estos solo alteran la geometría de las imágenes, representando la variedad de perspectiva, tamaño o posición que pueden poseer las flores en la vida real. Se excluyeron filtros de corrección de color, ya que el color de las flores es algo crucial en su clasificación y pueden afectar al resultado que se busca.

Esto genera nuevas imágenes que se suman al conjunto ya establecido de train, con la finalidad de enseñar de mejor manera al modelo. Estos métodos fueron aplicados en el código contenido en “Flores Data augmentation.ipynb”.

[Data Augmentation notebook](https://colab.research.google.com/drive/1A9FnXfncx8jqNY-BCPW2m2FVZmCh0fFB?usp=sharing)
