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

Esto genera nuevas imágenes que se suman al conjunto ya establecido de train, con la finalidad de enseñar de mejor manera al modelo. Estos métodos fueron aplicados en el código contenido en “Flores_Data_augmentation_and_first_model.ipynb".

[First model notebook](https://colab.research.google.com/drive/1A9FnXfncx8jqNY-BCPW2m2FVZmCh0fFB?usp=sharing)


## Modelos
## Primera iteración:
El primer modelo y sus resultados puede ser consultado en el archivo "Flores_Data_augmentation_and_first_model.ipynb" que se encuentra en este repositorio, o en el siguiente notebook:
[First model notebook](https://colab.research.google.com/drive/1A9FnXfncx8jqNY-BCPW2m2FVZmCh0fFB?usp=sharing)

Para el primer modelo se utilizó una Red Neuronal Convolucional (CNN) basándose en una arquitectura VGG, extraída de "Very Deep Convolutional Networks for Large-Scale Image Recognition" por Simonyan y Zisserman (2014).

El modelo es una versión muy simplicidad de la arquitectura de VGG. Ya que en ves de utilizar 64,128,256,512 y 512 canales, utiliza 32,64 y 64, esto se hizo con la finalidad de reducir la complejidad del modelo y poder hacer iteraciones con mayor facilidad. La descripción del modelo se encuentra a continuación:

- Input Shape, 120 x 120 pixeles y es rgb.
- Capa Conv2D con 32 filtros: Tamaño de kernel (3, 3), padding "same" y función de activación ReLU
- Capa Conv2D con 32 filtros: Tamaño de kernel (3, 3), padding "same" y función de activación ReLU
- Capa MaxPooling2D: Tamaño de pool (2, 2)
- Capa Conv2D con 64: filtros: Tamaño de kernel (3, 3), padding "same" y función de activación ReLU
- Capa Conv2D con 64: filtros: Tamaño de kernel (3, 3), padding "same" y función de activación ReLU
- Capa MaxPooling2D: Tamaño de pool (2, 2)
- Capa GlobalAveragePooling2D: Convierte las caracteristicas 2D a un vector plano.
- Capa Densa: Con 64 unidades y función de activación ReLU
- Capa Dropout: Con tasa de 0.5
- Capa Densa: Con 13 unidades y función de activación softmax

### Resultados:
La selección de métricas para la evaluación del modelo se basa en “An Introduction to ROC Analysis” de Fawcett (2005) el cual menciona que el “acurracy” no abarca toda la complejidad de problemas de clasificación, ya que puede existir diferentes orígenes de los errores o desbalanceo en los datos. Es por esto que el modelo se evaluó el modelo utilizando las siguientes métricas:

 - Precisión: Determina que tan acertado es el modelo cuando predice una "positive label", en el caso de la multiclase se determina que todas aquellas clases que no son la que corresponde son "negative, en vez de usar un enfoque binario como es el caso de usar solo dos clases.
 - Recall: De todas las "positive instances" que existen cuantas identificaste correctamente. Por ejemplo si hay 1000 imágenes de "california_poppy" cuantas detectaste.
 - F1-score: Es el balance entre las dos métricas anteriores.
 - Acurracy: Es la proporción de precciones correctas entre predicciones totales.

El soporte solo nos indica el número de ejemplos reales de la clase, no es una métrica.

#### Matriz de confusión:

<img width="791" height="704" alt="Captura de pantalla 2026-04-19 225206" src="https://github.com/user-attachments/assets/ca75b488-ac94-4382-9b82-aef014cf4bfb" />

  
| Clase | Precisión | Recall | F1-score | Soporte |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.64      | 0.62   | 0.63     | 200     |
| 1     | 0.56      | 0.33   | 0.42     | 196     |
| 2     | 0.53      | 0.42   | 0.47     | 205     |
| 3     | 0.65      | 0.61   | 0.63     | 196     |
| 4     | 0.47      | 0.40   | 0.43     | 210     |
| 5     | 0.50      | 0.71   | 0.59     | 194     |
| 6     | 0.69      | 0.62   | 0.65     | 211     |
| 7     | 0.78      | 0.81   | 0.79     | 211     |
| 8     | 0.52      | 0.72   | 0.60     | 210     |
| 9     | 0.59      | 0.63   | 0.61     | 214     |
| 10    | 0.55      | 0.80   | 0.65     | 206     |
| 11    | 0.64      | 0.58   | 0.61     | 210     |
| 12    | 0.54      | 0.36   | 0.43     | 197     |



| Métrica                  | Valor |
| ------------------------ | ----- |
| Accuracy                 | 0.59  |
| Macro Avg (Precision)    | 0.59  |
| Macro Avg (Recall)       | 0.58  |
| Macro Avg (F1-score)     | 0.58  |
| Weighted Avg (Precision) | 0.59  |
| Weighted Avg (Recall)    | 0.59  |
| Weighted Avg (F1-score)  | 0.58  |

### Conclusiones:

El “accuracy” del modelo es de 0.59, es decir que, aproximadamente, acierta cada 6 de 10 imágenes, un rendimiento intermedio, además el tanto el “recall”, “precision” y “F1” son prácticamente iguales lo que indica que el modelo no se encuentra fuertemente sesgado.	

La matriz de confusión muestra un buen desempeño con las clases 7, 8 y 10, no obstante, posee grandes confusiones con las clases 1,2 y 4, además de errores con la clase 12. Esto se puede deber a que las clases 7, 8 y 10 son de colores y formas diferentes a las otras clases, no obstante, la clase 1,2 y 4 tienen colores y formas muy similares, siendo todas ellas de un amarillo/naranja. 

En conclusión, el modelo se desempeña de buena manera en el aprendizaje de algunas clases, pero aquellas que son similares presentan un desafío para el modelo. Como mejora futura y con la finalidad de mejorar en estás clases problemáticas se propone el uso de un modelo más robusto más apegado a VGG11 o VGG16.

## Segunda iteración:
El modelo y sus resultados se pueden consultar en el siguiente notebook:
[Modelo VGG-11](florecitas-vgg-11.ipynb)

Asimismo, el modelo entrenado se encuentra disponible en:
[Modelo entrenado](mi_modelo%20(1).keras)

Si deseas correr el modelo usa:
[AppVGG11](app2.py)

Para la segunda iterración se implmento un VGG 11, quitando el últimmo bloque de 512, además de añadir batch normalization para estandarizar los datos dentro de la red y mejorar el aprendizaje del modelo. La descripción a detalle del modelo se presenta a continuación, así como una imagen del mismo:

<img width="566" height="89" alt="image" src="https://github.com/user-attachments/assets/faacc143-1a4a-4746-b0f4-ece1ac8bb88e" />


- Input Shape: Imágenes de 120 x 120 píxeles en formato RGB.
#### Bloque 1
- Capa Conv2D con 64 filtros: kernel (3,3), padding "same"
- BatchNormalization
- Activación ReLU
- Capa Conv2D con 64 filtros: kernel (3,3), padding "same"
- BatchNormalization
- Activación ReLU
- Capa MaxPooling2D: pool (2,2)
#### Bloque 2
- Capa Conv2D con 128 filtros: kernel (3,3), padding "same"
- BatchNormalization
Activación ReLU
- Capa Conv2D con 128 filtros: kernel (3,3), padding "same"
- BatchNormalization
- Activación ReLU
- Capa MaxPooling2D: pool (2,2)
#### Bloque 3
- Capa Conv2D con 256 filtros: kernel (3,3), padding "same"
- BatchNormalization
- Activación ReLU
- Capa Conv2D con 256 filtros: kernel (3,3), padding "same"
- BatchNormalization
- Activación ReLU
- Capa MaxPooling2D: pool (2,2)
#### Bloque 4
- Capa Conv2D con 512 filtros: kernel (3,3), padding "same"
- BatchNormalization
- Activación ReLU
- Capa Conv2D con 512 filtros: kernel (3,3), padding "same"
- BatchNormalization
- Activación ReLU
- Capa MaxPooling2D: pool (2,2)
#### Clasificador
- Capa GlobalAveragePooling2D: Convierte los mapas de características en un vector
- Capa Densa: 256 unidades con activación ReLU
- Capa Dropout: tasa de 0.5 (para reducir overfitting)
- Capa Densa final: 13 unidades con activación softmax (una por cada clase de flor)

### Resultados:
| Clase | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.94      | 0.91   | 0.92     | 200     |
| 1     | 0.69      | 0.73   | 0.71     | 196     |
| 2     | 0.85      | 0.78   | 0.81     | 205     |
| 3     | 0.93      | 0.68   | 0.79     | 196     |
| 4     | 1.00      | 0.09   | 0.16     | 210     |
| 5     | 1.00      | 0.10   | 0.18     | 194     |
| 6     | 0.97      | 0.91   | 0.94     | 211     |
| 7     | 1.00      | 0.55   | 0.71     | 211     |
| 8     | 0.88      | 0.65   | 0.75     | 210     |
| 9     | 0.92      | 0.49   | 0.64     | 200     |
| 10    | 0.47      | 1.00   | 0.64     | 206     |
| 11    | 0.99      | 0.85   | 0.92     | 210     |
| 12    | 0.29      | 1.00   | 0.45     | 197     |

| Métrica                  | Valor |
|--------------------------|-------|
| Accuracy                 | 0.67  |
| Macro Avg (Precision)    | 0.84  |
| Macro Avg (Recall)       | 0.67  |
| Macro Avg (F1-score)     | 0.66  |
| Weighted Avg (Precision) | 0.84  |
| Weighted Avg (Recall)    | 0.67  |
| Weighted Avg (F1-score)  | 0.66  |

#### Matriz de confusión:
<img width="799" height="719" alt="image" src="https://github.com/user-attachments/assets/2fbc96fb-504d-4f73-a82f-25f01baea9de" />

### Conclusiones:
El “accuracy” del modelo es de 0.67, es decir que, aproximadamente, acierta cada 7 de 10 imágenes, una mejora de 0.08 con respecto al modelo anterior. Por otra parte la preccision es alta 0.84, pero el recall 0.67 y f1 0.66  lo indica que el modelo no se encuentra bien balanceado y posee dificultades para indentificar todas las classes de manera correcta.

La matriz de confusión muestra de manera clara la confusión que existe con la clase 12 (water_lily), clase que tiene una precision de 0.29, y que no logra identificar de manera correcta la clase 4,5, esto se respalda por su bajo recall,0.09 y 0.10 ,y alta preccision mostrando que el modelo presenta mucho falsos positivos.

En conclusión, el modelo fue una mejora a la anterior versión, pero no obtuvo los resultados esperados, e incremento la confusión con la clase 12 (water_lily). Para la siguiente iteración se implementará un VGG1 con transfer learning, con la finalidad de visualizar la eficacia de está técnica. 



## Referencias
Fawcett, T. (2005). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861-874. https://doi.org/10.1016/j.patrec.2005.10.010

Ioffe, S., & Szegedy, C. (2015, 11 febrero). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv.org. https://arxiv.org/abs/1502.03167

Simonyan, K., & Zisserman, A. (2014, 4 septiembre). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv.org. https://arxiv.org/abs/1409.1556

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1), 1929–1958.
