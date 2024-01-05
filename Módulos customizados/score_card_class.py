import numpy as np
import pandas as pd

class score_card:
  def __init__(self):
    return None
  

  def generate_scorecard(self,model_coef,binning_df,features,B):
    
    """ Genera un DataFrame con la Scorecard

    Parameters:
    ----------
    model_coef : coeficientes de regresión logística
    binning_df: data frame agrupado por variable, Rango(string) y WOE correspondiente
    features: variables a considerar en la scorecard
    B: Valor de reescalamiento para la scorecard
    Returns:
    -------
    DataFrame agrupado por variable, Rango(string) y Score asociado
    """
    lst = []
    cols = ['Variable','Binning','Score']
    coef = model_coef[0]
    for i in range(len(features)):
        f = features[i]
        df = binning_df[binning_df['features']==f]
        for index,row in df.iterrows():
            lst.append([f,row['bin'],int(round(-coef[i]*row['woe']*B))])
    data = pd.DataFrame(lst, columns=cols)
    return data


# Fución para convertir en floats
  def str_to_int(self,s):
    if s == '-inf' or s==np.inf:
        return -999.0
    elif s=='inf' or s==-np.inf:
        return 999.0
    else:
        return float(s)

# Función para mapear valores de las variables a rangos
  def map_value_to_bin(self,feature_value,feature_to_bin):

    for idx, row in feature_to_bin.iterrows():
        bins = str(row['Binning'])
        left_open = bins[0]=="("
        right_open = bins[-1]==")"
        binnings = bins[1:-1].split(',')
        in_range = True
        # Revisión intervalo izquierdo:
        if left_open:
            if feature_value<= self.str_to_int(binnings[0]):
                in_range = False
        else:
            if feature_value< self.str_to_int(binnings[0]):
                in_range = False
        # Revisión intervalo derecho:
        if right_open:
            if feature_value>= self.str_to_int(binnings[1]):
                in_range = False
        else:
            if feature_value> self.str_to_int(binnings[1]):
                in_range = False
        if in_range:
            return row['Binning']
    return ""

# Función para calcular la suma de los scores de cada rango de las variables
  def map_to_score(self,df,score_card):

    scored_columns = list(score_card['Variable'].unique())
    score = 0
    for col in scored_columns:
        feature_to_bin = score_card[score_card['Variable']==col]
        feature_value = df[col] # Es una lista de valores
        selected_bin = self.map_value_to_bin(feature_value,feature_to_bin) # Es una lista de rangos string
        selected_record_in_scorecard = feature_to_bin[feature_to_bin['Binning'] == selected_bin]
        score += selected_record_in_scorecard['Score'].iloc[0]
    return score


  def calculate_score_with_card(self,df,score_card,A):

    df['score'] = df.apply(self.map_to_score,args=(score_card,),axis=1)
    df['score'] = df['score']+A
    df['score'] = df['score'].astype(int)
    return df
