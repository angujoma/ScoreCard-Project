# Desarrollo de punta a punta de una Scorecard de originación para créditos hipotecarios

## Fuentes de información
El conjunto de datos usado es público y se encuentra en: https://www.kaggle.com/competitions/home-credit-default-risk/data. 

Se considera información relativa a:
* Solicitudes
* Solcitudes previas
* Comportamiento interno de pagos
* Buró de Crédito
  
El diagrama de relación es el siguiente:

![diagrama tablas](https://github.com/angujoma/ScoreCard-Project/assets/141785336/7a08cff0-3d8a-487a-970c-3b3bec8382b6)

El diccionario de columnas se encuentra en este repositorio. 

## Predictores usados

* EXT_SOURCE_3: Score 1 Normalizado del cliente construido con información externa.
* EXT_SOURCE_2: Score 2 Normalizado del cliente construido con información externa.
* EXT_SOURCE_1: Score 3 Normalizado del cliente construido con información externa.
* REGION_RATING_CLIENT: Calificación interna del cliente.
* bd_CREDITS_ACTIVE_sum: Suma de los créditos activos del cliente en Buró al momento de la solicitud.
* pre_app_APP_REFUSED_sum: Cuantas solicitudes previas a la actual, fueron rechazadas al cliente.
* bd_CREDITS_CLOSED_sum: Suma de los créditos cerrados del cliente en Buró al momento de la solicitud. 
* REG_CITY_NOT_WORK_CITY: Indica si la dirección del trabajo del cliente es la proporcionada en el contrato. 
* SUM_FLAGS_DOCUMENTS: Cuantos documentos no obligatorios entregó el cliente al momento de la solicitud. 
* REG_CITY_NOT_LIVE_CITY: Indica si la dirección del cliente es la proporcionada en el contrato.

## Métricas de desempeño obtenidas:
* Entrenamiento:
   * Accuracy: 0.877
   * RoC AuC: 0.721
   * GINI: 0.442
   * KS:0.32
* Test:
   * Accuracy: 0.878
   * RoC AuC: 0.729
   * GINI: 0.457
   * KS: 0.34



## Descripción de los archivos:
* **Deployment**: Carpeta que contiene los archivos necesarios para ejecutar la aplicación localmente.
* **Módulos customizados**: Contiene los módulos para construcción de la Scorecard y modelado.
* **HCDR_model.pkl**: Archivo ejecutable del modelo de regresión logisítica.
* **Ingenería de variables.ipynb**: Notebook con toda la ingenería de variables, cruces de información y limpieza.
* **Modelado.ipynb**: Notebook con todo el modelado y construcción de la Scorecard.
* **Imput_train_values.pkl**: Valores usados en la imputación de valores faltantes.
* **Predictors**: Predictores finales usados en el modelo. 


      
  


