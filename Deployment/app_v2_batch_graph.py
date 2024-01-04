import numpy as np
import pandas as pd
import pickle
from score_card_class import score_card
import streamlit as st
import matplotlib.pyplot as plt

map_to_woe=pd.read_pickle("map_feat_bin_woe.pkl")
scorecard=pd.read_pickle("score_card.pkl")
A =600

gsc=score_card()


def main():
    st.title("Cálculo de Score de originación")
    st.subheader("Por Ángel Gustavo José Martínez.")
    st.divider()
    
    file_uploaded = st.file_uploader("Carga el archivo con la información necesaria de los clientes para calcular su score:", type=["csv","txt"])
    if file_uploaded is not None:
        df=pd.read_csv(file_uploaded,header=None)
        new_cols={0:"EXT_SOURCE_3",
                    1:"EXT_SOURCE_2",
                    2:"EXT_SOURCE_1",
                    3:"REGION_RATING_CLIENT",
                    4:"bd_CREDITS_ACTIVE_sum",
                    5:"bd_CREDITS_CLOSED_sum", 
                    6:"pre_app_APP_REFUSED_sum",
                    7:"REG_CITY_NOT_WORK_CITY",
                    8:"SUM_FLAGS_DOCUMENTS",
                    9:"REG_CITY_NOT_LIVE_CITY"}
        df=df.rename(columns=new_cols)
        
        df_prediction=gsc.calculate_score_with_card(df,scorecard,A)
        df_prediction["Resultado"]=df_prediction["score"].apply(lambda x: "Aceptado" if x>655 else "Rechazado")

        from matplotlib.offsetbox import Bbox
        def addlabels(x,y):
            for i in range(len(x)):
                plt.text(i, y[i], y[i], ha = 'center')
       
        a=pd.DataFrame(df_prediction["Resultado"].value_counts()).reset_index()
        a.rename(columns={"Solicitudes":"Resultado","count":"Solicitudes"},inplace=True)
        #Cálculos de resumen:
        sol_tot=df.shape[1]
        p_score_r=df[df["Resultado"]=="Rechazado"]["score"].mean()
        p_score_a=df[df["Resultado"]=="Aceptado"]["score"].mean()
        tasa_rechazo=df[df["Resultado"]=="Rechazado"]["score"].count()/df.shape[1]
        tasa_aceptacion=df[df["Resultado"]=="Aceptado"]["score"].count()/df.shape[1]
        textstr = '\n'.join((
        r'* Solicitudes=%.0f' % (sol_tot, ),
        r'* Score prom. Rech.=%.2f' % (p_score_r, ),
        r'* Score prom. Acep.=%.2f' % (p_score_a, ),
        r'* Tasa de Rechazo=%.2f%%' % (tasa_rechazo, ),
        r'* Tasa de Aceptación=%.2f%%' % (tasa_aceptacion, )
         ))
   
        
        
    
    
    html_temp = """ 
    <div style ="background-color:white;padding:5px"> 
    <h2 style ="color:black;text-align:left;">Resultado: </h2> 
    </div> 
    """
      

    st.markdown(html_temp, unsafe_allow_html = True) 
      
   
    result ="" 
    
    
    if st.button("Predict"): 
        result = st.write(df_prediction)
        
    
   #GRÁFICO:
        colores=["firebrick","cornflowerblue"]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(a["Resultado"],a["Solicitudes"],color=colores)
        ax.text("Aceptado",4 ,textstr,fontsize = 11,
        bbox = dict(facecolor = 'grey', alpha = 0.2))
        addlabels(a["Resultado"],a["Solicitudes"])
        plt.title("Resumen de Originación")
        plt.xlabel("Resultado")
        plt.ylabel("Solicitudes")
     
        st.pyplot(fig)

if __name__=='__main__': 
    main() 



