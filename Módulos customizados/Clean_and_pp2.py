# Función completitud de columnas:
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve,auc,recall_score,accuracy_score,precision_score,f1_score,cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import math


def completitud(df):
    """Función que calcula el número de registros
    nulos por columna en un DataFrame.

    Parameters:
    ----------
    df : DataFrame

    Returns
    -------
    Dataframe:
    """
    comp=pd.DataFrame(df.isnull().sum())
    comp.reset_index(inplace=True)
    comp=comp.rename(columns={"index":"Columna",0:"Total_reg_nulos"})
    comp["Per_completitud"]=(1-comp["Total_reg_nulos"]/df.shape[0])*100
    comp=comp.sort_values(by="Per_completitud",ascending=True)
    #comp.reset_index(drop=True,inplace=True)
    return comp

def imput_values_prev(df):
    
    """Función para imputar valores antes del procesamiento de la data.
    
    Parameters:
    ----------
    df : DataFrame

    Returns
    -------
    Dataframe imputado.:
    """
    
    for f in ['EXT_SOURCE_1',"EXT_SOURCE_2",'EXT_SOURCE_3',"cc_SK_DPD_max","cc_CARDS_APROVED_sum",
             'cc_CARDS_REFUSED_sum',"cc_AMT_CREDIT_LIMIT_ACTUAL_max"]:
             df[f].fillna(-1,inplace=True)
        
        

#Función para imputar valores faltantes en train y test.
def imput_values(df):
    
    """Función para imputar valores de acuerdo al data set de HCR

    Parameters:
    ----------
    df : DataFrame

    Returns
    -------
    Dataframe imputado y columnas con el valor usado para imputar:
    """
    list_imputs_values=[]
    feat_nulls=completitud(df)
    for c in feat_nulls[feat_nulls["Per_completitud"]<100]["Columna"].tolist():

        if c.startswith(("bd_","AMT_REQ_")or c in ['EXT_SOURCE_1',"EXT_SOURCE_2",'EXT_SOURCE_3',"cc_SK_DPD_max",
                                                 "cc_CARDS_APROVED_sum",'cc_CARDS_REFUSED_sum',"cc_AMT_CREDIT_LIMIT_ACTUAL_max"] ):
            imputer = SimpleImputer(missing_values = np.nan, strategy ='constant', fill_value=-1)
        else:
            imputer= SimpleImputer(missing_values = np.nan, strategy ='mean')
        imputer.fit(df[[c]])
        df[[c]]=imputer.transform(df[[c]])
        list_imputs_values.append(str(c)+","+str(imputer.statistics_[0]))
    val_inputs_df=pd.DataFrame(list_imputs_values,columns=["var"])
    val_inputs_df=val_inputs_df["var"].str.split(",",expand=True)
    val_inputs_df.columns=["variable","valor_imputado"]
    return val_inputs_df

# Función para calcular el IV:
def cal_IV(df, feature, target):
    """ Calcula el valor de la información (Information Value) dada una variable.

    Parameters:
    ----------
    df : DataFrame
    feature: Variable a la cual se le quiere calcular el IV
    target: Variable objetivo
    Returns:
    -------
    Dataframe con el nombre de la variable y el calculo del IV
    """
    lst = []
    cols=['Variable', 'Value', 'All', 'Bad']
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])
    #print(lst)
    data = pd.DataFrame(lst, columns=cols)
    data = data[data['Bad'] > 0]
    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Bad'] / data['Distribution Good'])
    data['IV'] = (data['WoE'] * (data['Distribution Bad'] - data['Distribution Good'])).sum()
    data = data.sort_values(by=['Variable', 'Value'], ascending=True)
    return data[["Variable","IV"]].drop_duplicates()

# Función para calcular el WOE
def cal_WOE(df,features,target):
    """ Calcula el peso de la evidencia WOE (Wieight of Evidence) dada una lista de variables.

    Parameters:
    ----------
    df : DataFrame
    features: lista de variables
    target: Variable objetivo
    Returns
    -------
    Dataframe con las variables originales más variables con su respectivo WOE
    """
    df_new = df.copy()
    for f in features:
        df_woe = df_new.groupby(f).agg({target:['sum','count']})
        df_woe.columns = list(map(''.join, df_woe.columns.values))
        df_woe = df_woe.reset_index()
        df_woe = df_woe.rename(columns = {target+'sum':'bad'})
        df_woe = df_woe.rename(columns = {target+'count':'all'})
        df_woe['good'] = df_woe['all']-df_woe['bad']
        df_woe = df_woe[[f,'good','bad']]
        df_woe['bad_rate'] = df_woe['bad'].mask(df_woe['bad']==0, 1)/df_woe['bad'].sum() # ocultamos a 1 para evitar log(0)
        df_woe['good_rate'] = df_woe['good']/df_woe['good'].sum()

        df_woe['woe'] = np.log(df_woe['bad_rate'].divide(df_woe['good_rate'],fill_value=1))
        df_woe.columns = [c if c==f else c+'_'+f for c in list(df_woe.columns.values)] # Para identificar de que feature viene se re etiqueta
        df_new = df_new.merge(df_woe,on=f,how='left')
    return df_new

#Esta función sirve para mapear los rangos y su correspondiente WoE calculado en el conjunto de Test.
def cal_woe_in_validation(df_woe,df_test,cols):
    
    for c in cols:
        df2=df_woe[df_woe["features"]==c][["bin","woe"]]
        replaces=dict(df2.values)
        df_test['woe_'+c+'_bin']=df_test[c+'_bin'].map(replaces)
        
        

def plot_dist_mora(df,list_vars,target):
    i=0
    plt.figure()
    fig, ax = plt.subplots(3,1,figsize=(25,20)) #3,1

    for feat in list_vars:
        aux=df[[feat,target]].copy()
        aux['n']=1
        aux = aux.pivot_table(columns=target,
                            index=feat,
                            values='n',
                            aggfunc='count')
        aux["Tot"]=aux[1]+aux[0]
        aux["perc_mora"]=(aux[1]/aux["Tot"])
        #fig = plt.figure()
        i += 1
        plt.subplot(5,3,i)
        ax = aux['Tot'].plot(kind='bar',grid=True,color='orange') #Pastel1
        ax2 = ax.twinx()
        ax2.plot(aux['perc_mora'].values, linestyle='-', linewidth=2.0,color='red')
        plt.xlabel('x')
        plt.ylabel('%mora')
        plt.title("% de mora por rango de  "+feat,color='red',fontsize=20)
        locs, labels = plt.xticks()
    #plt.tick_params(axis='both', which='major', labelsize=5)
    fig.tight_layout()
    plt.show();
        


def plot_feature_importance(X,data_arg,model):
    
    if data_arg==True:
        
        plt.figure(figsize = (25,16))
        importances = pd.DataFrame({'feature':X.columns,'importance':np.abs(model.named_steps["model"].coef_[0])})
        importances = importances.sort_values('importance',ascending=False)
        g = sns.barplot(x='importance', y='feature', data=importances,
                  saturation=0.8, label="Total")
        g.set(xlabel='Importancia', ylabel='Predictor', title='Importancia de los predictores Regresión Logística')

    else:
        
        plt.figure(figsize = (10,10))
        importances = pd.DataFrame({'feature':X.columns,'importance':np.abs(model.coef_[0])})
        importances = importances.sort_values('importance',ascending=False)
        g = sns.barplot(x='importance', y='feature', data=importances,
                  saturation=0.8, label="Total")
        g.set(xlabel='Importancia', ylabel='Predictor', title='Importancia de los predictores Regresión Logística')

#Función para obtener el top n de las variables más importantes.
def get_feat_imp(model,data_arg,n_features,X):
    
    if data_arg==True:
        importances = pd.DataFrame({'feature':X.columns,'importance':np.abs(model.named_steps["model"].coef_[0])})
        importances = importances.sort_values('importance',ascending=False)

    else:
        importances = pd.DataFrame({'feature':X.columns,'importance':np.abs(model.coef_[0])})
        importances = importances.sort_values('importance',ascending=False)

    return importances.head(n_features).reset_index(drop=True)


# Función para plotear metricas, importancia de las variables, matriz de confución y curva RoC
def plot_metrics(y,y_pred,model_probs): #Corregir por que no se ocupa el parámetro X
    
    """Calcula la precision,accuracy,recall,f1 score, ROC-AUC, Kappa Score además de graficar
    a importancia de las variables, la curva ROC y la matriz de confusión.
    Parameters:
    ----------
    Y : Valor conocido del target
    Y_pred: Valor predicho del modelo
    model_probs: Probabilidad predicha del modelo
    model_log: Modelo a ajustar
    data_arg: Valor booleano que indica si se aplica data argumentation mediante un Pipeline
    Returns:
    Impresion en pantalla de las métricas y gráficos."""

    print('---' * 45)
    print('MÉTRICAS DE DESEMPEÑO DEL CLASIFICADOR:')
    print('---' * 45)
    random_probs = [0 for _ in range(len(y))]

    prec=precision_score(y, y_pred)
    model_auc = roc_auc_score(y, model_probs)
    rec=recall_score(y, y_pred)
    acc=accuracy_score(y, y_pred)
    f1=f1_score(y, y_pred)
    kappa=cohen_kappa_score(y, y_pred)

    print('PRECISION=%.3f' % (prec))
    print('ACCURACY=%.3f' % (acc))
    print('RECALL=%.3f' % (rec))
    print('F1 SCORE=%.3f' % (f1))
    print('ROC AUC=%.3f' % (model_auc))
    print('GINI=%.3f' % ((2*model_auc)-1))
    print('KAPPA SCORE=%.3f' % (kappa))

    # Curva ROC
    random_fpr, random_tpr, _ = roc_curve(y, random_probs)
    model_fpr, model_tpr, _ = roc_curve(y, model_probs)
    figure, axis = plt.subplots(1,1, figsize=(4, 3),squeeze=False)
    axis[0, 0].plot(random_fpr, random_tpr, linestyle='--', label='Random',color="red")
    axis[0, 0].plot(model_fpr, model_tpr, marker='.', label='Model',color='orange')
    plt.title("Curva ROC en datos de validación")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # Matriz de confusión
    plt.figure(figsize = (4,3))
    sns.heatmap(confusion_matrix(y, y_pred),fmt='g',annot=True)
    plt.title('Matriz de Confusión', y=1.1)
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')

# Función para realizar GridSearch
def grid_search(X_train,y_train,estimator,param_grid):
    
    """ Realiza el algoritmo de Grid Search y validación cruzada.

    Parameters:
    ----------
    X_train : Datos de entrenamiento
    y_train: Target de entrenamiento
    estimator: Clasificador a ajustar o pipeline
    param_grid: Lista de paramétros
    Returns:
    -------
    El clasificador con el mejor scoring definido de acuerdo a la lista de paramétros ingresados.
    """
    grid_search = GridSearchCV(cv=StratifiedKFold(10),
                  verbose=True,
                  scoring='accuracy',
                  estimator=estimator,
                  n_jobs=-1,
                  param_grid=param_grid)

    grid_search.fit(X_train,y_train)
    print(f"Best Score : {grid_search.best_score_}")
    print(f"Best Params : {grid_search.best_params_}")
    return grid_search.best_estimator_

#Creación de rangos sobre variables especificas:
def var_to_rank(df):
    
    EXT_SOURCE_2_bins=[-math.inf,0,0.3,0.5,0.7,math.inf]
    df["EXT_SOURCE_2_bin"]=pd.cut(df['EXT_SOURCE_2'],bins=EXT_SOURCE_2_bins,right=False).astype(str)
  #------------------------------------------------------------------------------------------------
    EXT_SOURCE_3_bins=[-math.inf,0,0.3,0.5,0.7,math.inf]
    df["EXT_SOURCE_3_bin"]=pd.cut(df['EXT_SOURCE_3'],bins=EXT_SOURCE_3_bins,right=False).astype(str)
  #------------------------------------------------------------------------------------------------
    SUM_FLAGS_DOCUMENTS_bins=[0,1,2,3,5,math.inf]
    df["SUM_FLAGS_DOCUMENTS_bin"]=pd.cut(df['SUM_FLAGS_DOCUMENTS'],bins=SUM_FLAGS_DOCUMENTS_bins,right=False).astype(str)
  #------------------------------------------------------------------------------------------------
  #'REG_CITY_NOT_LIVE_CITY'
    REG_CITY_NOT_LIVE_CITY_bins=[-math.inf,0,1,math.inf]
    df['REG_CITY_NOT_LIVE_CITY_bin']=pd.cut(df['REG_CITY_NOT_LIVE_CITY'],bins=REG_CITY_NOT_LIVE_CITY_bins,right=True).astype(str)
  #df.rename(columns={'REG_CITY_NOT_LIVE_CITY':'REG_CITY_NOT_LIVE_CITY_bin'},inplace=True)
  #------------------------------------------------------------------------------------------------
    DEF_30_CNT_SOCIAL_CIRCLE_bins=[0,1,2,3,5,math.inf]
    df["DEF_30_CNT_SOCIAL_CIRCLE_bin"]=pd.cut(df["DEF_30_CNT_SOCIAL_CIRCLE"],bins=DEF_30_CNT_SOCIAL_CIRCLE_bins,right=False).astype(str)
  #------------------------------------------------------------------------------------------------
    EXT_SOURCE_1_bins=[-math.inf,0,0.3,0.6,1,math.inf]
    df["EXT_SOURCE_1_bin"]=pd.cut(df["EXT_SOURCE_1"],bins=EXT_SOURCE_1_bins,right=False).astype(str)
  #------------------------------------------------------------------------------------------------
  #'REG_REGION_NOT_LIVE_REGION'
    REG_REGION_NOT_LIVE_REGION_bins=[-math.inf,0,1]
    df['REG_REGION_NOT_LIVE_REGION_bin']=pd.cut(df['REG_REGION_NOT_LIVE_REGION'],bins=REG_REGION_NOT_LIVE_REGION_bins,right=True)
  #df.rename(columns={'REG_REGION_NOT_LIVE_REGION':'REG_REGION_NOT_LIVE_REGION_bin'},inplace=True)
  #------------------------------------------------------------------------------------------------
  #'REGION_RATING_CLIENT'
    REGION_RATING_CLIENT_bins=[1,2,3,math.inf]
    df['REGION_RATING_CLIENT_bin']=pd.cut(df['REGION_RATING_CLIENT'],bins=REGION_RATING_CLIENT_bins,right=False)
  #df.rename(columns={'REGION_RATING_CLIENT':'REGION_RATING_CLIENT_bin'},inplace=True)
  #------------------------------------------------------------------------------------------------
    bd_MICRO_LOANS_sum_bins=[-math.inf,0,4,6,math.inf]
    df["bd_MICRO_LOANS_sum_bin"]=pd.cut(df["bd_MICRO_LOANS_sum"],bins=bd_MICRO_LOANS_sum_bins,right=False).astype(str)
  #------------------------------------------------------------------------------------------------
    AMT_REQ_CREDIT_BUREAU_DAY_bins=[-math.inf,0,4,math.inf]
    df["AMT_REQ_CREDIT_BUREAU_DAY_bin"]=pd.cut(df["AMT_REQ_CREDIT_BUREAU_DAY"],bins=AMT_REQ_CREDIT_BUREAU_DAY_bins,right=False).astype(str)
  #------------------------------------------------------------------------------------------------
    bd_CREDITS_ACTIVE_sum_bins=[-math.inf,0,3,6,math.inf]
    df["bd_CREDITS_ACTIVE_sum_bin"]=pd.cut(df["bd_CREDITS_ACTIVE_sum"],bins=bd_CREDITS_ACTIVE_sum_bins,right=False).astype(str)
  # ------------------------------------------------------------------------------------------------
    bd_bb_31_60PD_max_sum_bins=[-math.inf,0,2,4,math.inf]
    df["bd_bb_31_60PD_max_sum_bin"]=pd.cut(df["bd_bb_31_60PD_max_sum"],bins=bd_bb_31_60PD_max_sum_bins,right=False).astype(str)
  # ------------------------------------------------------------------------------------------------
    pre_app_APP_REFUSED_sum_bins=[0,3,8,10,math.inf]
    df["pre_app_APP_REFUSED_sum_bin"]=pd.cut(df["pre_app_APP_REFUSED_sum"],bins=pre_app_APP_REFUSED_sum_bins,right=False).astype(str)
  # ------------------------------------------------------------------------------------------------
    cc_CARDS_REFUSED_sum_bins=[-math.inf,1,4,math.inf]
    df["cc_CARDS_REFUSED_sum_bin"]=pd.cut(df["cc_CARDS_REFUSED_sum"],bins=cc_CARDS_REFUSED_sum_bins,right=False).astype(str)
  # ------------------------------------------------------------------------------------------------
    bd_CAR_LOANS_sum_bins=[-math.inf,2,4,math.inf]
    df["bd_CAR_LOANS_sum_bin"]=pd.cut(df["bd_CAR_LOANS_sum"],bins=bd_CAR_LOANS_sum_bins,right=False).astype(str)
  # ------------------------------------------------------------------------------------------------
    pre_app_POS_PORT_sum_bins=[0,2,4,9,math.inf]
    df["pre_app_POS_PORT_sum_bin"]=pd.cut(df["pre_app_POS_PORT_sum"],bins=pre_app_POS_PORT_sum_bins,right=False).astype(str)
  # ------------------------------------------------------------------------------------------------
  #REG_REGION_NOT_WORK_REGION
    REG_REGION_NOT_WORK_REGION_bins=[-math.inf,0,1]
    df['REG_REGION_NOT_WORK_REGION_bin']=pd.cut(df["REG_REGION_NOT_WORK_REGION"],bins=REG_REGION_NOT_WORK_REGION_bins,right=True)
  #df.rename(columns={'REG_REGION_NOT_WORK_REGION':'REG_REGION_NOT_WORK_REGION_bin'},inplace=True)
  #-------------------------------------------------------------------------------------------------
  #LIVE_CITY_NOT_WORK_CITY
    LIVE_CITY_NOT_WORK_CITY_bins=[-math.inf,0,1]
    df['LIVE_CITY_NOT_WORK_CITY_bin']=pd.cut(df["LIVE_CITY_NOT_WORK_CITY"],bins=LIVE_CITY_NOT_WORK_CITY_bins,right=True)
  #df.rename(columns={'LIVE_CITY_NOT_WORK_CITY':'LIVE_CITY_NOT_WORK_CITY_bin'},inplace=True)
  #-------------------------------------------------------------------------------------------------
    AMT_REQ_CREDIT_BUREAU_QRT_bins=[-math.inf,0,2,4,9,math.inf]
    df["AMT_REQ_CREDIT_BUREAU_QRT_bin"]=pd.cut(df["AMT_REQ_CREDIT_BUREAU_QRT"],bins=AMT_REQ_CREDIT_BUREAU_QRT_bins,right=False).astype(str)
  #-------------------------------------------------------------------------------------------------
    bd_bb_120PD_MORE_max_sum_bins=[-math.inf,0,2,4,9,math.inf]
    df["bd_bb_120PD_MORE_max_sum_bin"]=pd.cut(df["bd_bb_120PD_MORE_max_sum"],bins=bd_bb_120PD_MORE_max_sum_bins,right=False).astype(str)
  #-------------------------------------------------------------------------------------------------
  #'REG_CITY_NOT_WORK_CITY'
    REG_CITY_NOT_WORK_CITY_bins=[-math.inf,0,1]
    df['REG_CITY_NOT_WORK_CITY_bin']=pd.cut(df['REG_CITY_NOT_WORK_CITY'],bins=REG_CITY_NOT_WORK_CITY_bins,right=True)
  #df.rename(columns={'REG_CITY_NOT_WORK_CITY':'REG_CITY_NOT_WORK_CITY_bin'},inplace=True)
  #-------------------------------------------------------------------------------------------------
    bd_CREDITS_CLOSED_sum_bins=[-math.inf,0,3,5,math.inf]
    df["bd_CREDITS_CLOSED_sum_bin"]=pd.cut(df["bd_CREDITS_CLOSED_sum"],bins=bd_CREDITS_CLOSED_sum_bins,right=True).astype(str)
  #-------------------------------------------------------------------------------------------------
    bd_CREDITS_CARDS_sum_bins=[-math.inf,0,2,3,math.inf]
    df["bd_CREDITS_CARDS_sum_bin"]=pd.cut(df["bd_CREDITS_CARDS_sum"],bins=bd_CREDITS_CARDS_sum_bins,right=True).astype(str)
  #-------------------------------------------------------------------------------------------------
    pre_app_CASH_PORT_sum_bins=[0,2,5,math.inf]
    df["pre_app_CASH_PORT_sum_bin"]=pd.cut(df["pre_app_CASH_PORT_sum"],bins=pre_app_CASH_PORT_sum_bins,right=False).astype(str)
  #-------------------------------------------------------------------------------------------------
    CNT_CHILDREN_bins=[0,1,2,math.inf]
    df["CNT_CHILDREN_bin"]=pd.cut(df["CNT_CHILDREN"],bins=CNT_CHILDREN_bins,right=False).astype(str)

def score_distribution(data,x,target,bins,title):
  
  fig, ax = plt.subplots(figsize=(5, 3))
  sns.histplot(data=data, x=data[x], hue=target, stat='count',bins=bins,kde=True)
  ax.set_title(title)


def plot_KS(data_split,split,target,X,model):
  df=data_split.copy()
  df["proba"]=model.predict_proba(data_split[X])[:,1]
  df["y_hat"]=model.predict(data_split[X])
  df=df[["TARGET","proba","y_hat"]].sort_values('proba').reset_index(drop=True)

  df["Población_acumulada"]=df.index+1
  df["Malos acumulados"]=df[target].cumsum()
  df["Buenos acumulados"]=df["Población_acumulada"]-df[target].cumsum()
  df["Perc_Pob_acum"]=(df["Población_acumulada"])/(df.shape[0])
  df["Perc_Malos_acum"]=(df["Malos acumulados"])/(df[target].sum())
  df["Perc_Buenos_acum"]=(df["Buenos acumulados"])/(df.shape[0]-df[target].sum())   

  KS = max(df['Perc_Buenos_acum'] - df['Perc_Malos_acum'])
  print("El estadístico KS en",split,"es:", KS)
  
  plt.subplots(figsize=(5, 3))
  # Plot KS
  plt.plot(df['proba'], df['Perc_Buenos_acum'], color = 'r')
  # We plot the predicted (estimated) probabilities along the x-axis and the cumulative percentage 'bad' along the y-axis,
  # colored in red.
  plt.plot(df['proba'], df['Perc_Malos_acum'], color = 'b')
  # We plot the predicted (estimated) probabilities along the x-axis and the cumulative percentage 'good' along the y-axis,
  # colored in red.
  plt.xlabel('Probabilidad estimada de ser cliente Bueno')
  # We name the x-axis "Estimated Probability for being Good".
  plt.ylabel('Probabilidad acumulada %')
  # We name the y-axis "Cumulative %".
  plt.title('Estadístico Kolmogorov-Smirnov')
  # We name the graph "Kolmogorov-Smirnov".
  
def table_BR(data,bins_score,righ_side_score,col_score,target):
  data["Rank_score"]=pd.cut(data["score"],bins=bins_score,right=False)
  t=data.groupby("Rank_score").agg({"REGION_RATING_CLIENT":"count","TARGET":"sum"})
  t["BAD_RATE"]=t["TARGET"]/t["REGION_RATING_CLIENT"] 
  display(t.reset_index(drop=False))



