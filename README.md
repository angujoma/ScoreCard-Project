# Desarrollo de punta a punta de un ScoreCard de originación
La implementación de Scorecards en la industria financiera lleva bastante tiempo en uso y aún es vigente, pues a pesar de la existencia y posibilidad de implementar clasificadores más complejos como *Random Forest, SVM, AdaBoost, Gradient Boosting, XGboost o incluso Redes Neuronales*, no son de fácil interpretación (cajas negras), sobre todo para las áreas comerciales, además de que sulen consumir mucho más recursos computacionales sin importar si el modelo está implementado en nube (la facturación será más costosa).
* **Ventas de la Scorecard**:
    * Mayor intrpretabilidad.
    * Facilidad de implementación y menor gasto computacional.
    * Los cambios que se requieran en producción 
   
* **Desventajas**:
    * En algunos casos menor precisión que los algoritmos arriba mencionados.

 La escorecard desarrollada en el presente trabajo es para un **score de originación** para prestamos hipótecarios, cuya fuente de información usada se encuentra en: https://www.kaggle.com/competitions/home-credit-default-risk/data.

Se considera información interna de:
* Solicitudes
* Solcitudes previas
* Comportamiento de pagos

Información externa de:

* Buró de Crédito
* Comportamiento de pagos en otras insitituciones

El diagrama de como se relacionan los conjuntos de datos es el siguiente:

![ScoreCard-Project](diagrama tablas.png)
diagrama tablas.png
