# Desarrollo de punta a punta de un ScoreCard de originación
La implementación de Scorecards en la industria financiera lleva bastante tiempo en uso, pues a pesar de la existencia y posibilidad de implementar clasificadores más complejos como *Random Forest, SVM, AdaBoost, Gradient Boosting, XGboost o incluso Redes Neuronales*, no son de fácil interpretación (cajas negras), sobre todo para las áreas comerciales. También sulen consumir más recursos computacionales sin importar si el modelo está implementado en nube (la facturación será más costosa).
* **Ventas de la Scorecard**:
    * Mayor intrpretabilidad
    * Facilidad de implementación y menor gasto computacional
    * Los cambios o ajustes que se requieran en producción, son fáciles de realizar
   
* **Desventajas**:
    * En algunos casos menor precisión que los algoritmos mencionados.
 
## Fuente de información

 La Scorecard desarrollada en el presente trabajo es para **la originación** de prestamos hipótecarios, cuya fuente de información usada se encuentra en: https://www.kaggle.com/competitions/home-credit-default-risk/data.

Dicha fuente contiene información relativa a:
* Solicitudes
* Solcitudes previas
* Comportamiento interno de pagos
* Buró de Crédito
  
El diagrama de relación es el siguiente:

![diagrama tablas](https://github.com/angujoma/ScoreCard-Project/assets/141785336/7a08cff0-3d8a-487a-970c-3b3bec8382b6)
El diccionario de columnas se encuentra en en este repositorio. 

### Consideraciones:
* 

