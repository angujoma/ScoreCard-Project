# Desarrollo de punta a punta de un ScoreCard de originación
La implementación de Scorecards en la industria financiera lleva bastante tiempo en uso, pues a pesar de la existencia y posibilidad de implementar clasificadores más complejos como *Random Forest, SVM, AdaBoost, Gradient Boosting, XGboost o incluso Redes Neuronales*, no son de fácil interpretación (cajas negras), sobre todo para las áreas comerciales, además sulen consumir más recursos computacionales sin importar si el modelo está implementado en nube (la facturación es más costosa).
* **Ventas de la Scorecard**:
    * Mayor intrpretabilidad.
    * Facilidad de implementación y menor gasto computacional.
    * Los cambios que se requieran en producción son fáciles de realizar.
   
* **Desventajas**:
    * En algunos casos menor precisión que los algoritmos mencionados.

 La escorecard desarrollada en el presente trabajo es para **la originación** de prestamos hipótecarios, cuya fuente de información usada se encuentra en: https://www.kaggle.com/competitions/home-credit-default-risk/data.

Se considera información interna de:
* Solicitudes
* Solcitudes previas
* Comportamiento de pagos

Información externa de:

* Buró de Crédito
* Comportamiento de pagos en otras insitituciones

El diagrama de como se relacionan los conjuntos de datos es el siguiente:

![diagrama tablas](https://github.com/angujoma/ScoreCard-Project/assets/141785336/7a08cff0-3d8a-487a-970c-3b3bec8382b6)
