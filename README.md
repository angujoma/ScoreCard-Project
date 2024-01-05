# Desarrollo de punta a punta de una Scorecard de originación
La implementación de Scorecards en la industria financiera lleva bastante tiempo en uso, pues a pesar de la existencia y posibilidad de implementar clasificadores más complejos como *Random Forest, SVM, AdaBoost, Gradient Boosting, XGboost o incluso Redes Neuronales*, no son de fácil interpretación (cajas negras), sobre todo para las áreas comerciales. También sulen consumir más recursos computacionales, sin importar si el modelo está implementado en nube (la facturación será más costosa).
* **Ventas de la Scorecard**:
    * Mayor intrpretabilidad
    * Facilidad de implementación y menor gasto computacional
    * Los cambios o ajustes que se requieran en producción, son fáciles de realizar
   
* **Desventajas**:
    * En algunos casos menor precisión que los algoritmos mencionados.
      
La Scorecard desarrollada en el presente trabajo es para **la originación de prestamos hipótecarios**. Sin embargo, la metodología seguida también es aplicable a la construcción de scores de comportamiento, cobranza, propensión, fraudes, satisfacción, retención, etc. 

La última parte de este proyecto muestra la puesta en producción de la Scorecard de forma local.

## Fuentes de información
El conjunto de datos usado es público y se encuentra en: https://www.kaggle.com/competitions/home-credit-default-risk/data. 

Se considera información relativa a:
* Solicitudes
* Solcitudes previas
* Comportamiento interno de pagos
* Buró de Crédito
  
El diagrama de relación es el siguiente:

![diagrama tablas](https://github.com/angujoma/ScoreCard-Project/assets/141785336/7a08cff0-3d8a-487a-970c-3b3bec8382b6)

El diccionario de columnas se encuentra en en este repositorio. 

### Consideraciones:
* La target ya se encuentra definida y por tanto, la definición de cliente bueno y malo. Sin embargo, es importante mencionar que se deben definir **mediante matrices de transición** y en consenso con los **stakeholders**. Además de la ventana de tiempo en la que se considere la información.

## Desafíos del proyecto:
* Se tiene un desbalance en la distribución del target (8% Ctes buenos vs 92% ctes malos).
    * **Solución**: Se aplican dos técnicas de remuestreo: SMOTE (*Synthetic Minority Oversampling Technique*) y Undersampling, ambas combinanadas en un pipeline.

* El conjunto de datos después de realizar ingeniería de características y cruces entre tablas,  cuenta con **220 columnas** y 3 millones de registros, lo cual conlleva a tener un problema de dimensionalidad.
    * **Solución**: Se aplican filtros de valores faltantes, correlación, **regularización $L1$** y Valor de la información (IV).
    
      **El resultado de aplicar estas técnicas es la selección del top 10 de características con más poder predictivo para el modelo.**

## Resumen variables usadas y métricas obtenidas


## Descripción de los archivos:
* **Deplyment**: Carpeta que contiene los archivos necesarios para ejecutar la aplicación localmente.
* **Módulos customizados**: Contiene los módulos para construcción de la Scorecard y modelado.
* **HCDR_model.pkl**: Archivo ejecutable del modelo de regresión logisítica.
* **Ingeneria de variables**


      
  


