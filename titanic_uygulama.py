# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:29:30 2022

@author: User
"""

import streamlit as st
import pandas as pd
import numpy as np


def main():
    st.title("TİTANİC VERİ SETİ ML MODELİNİN UYGULANMASI")
    with st.beta_expander ("Sistem Açıklaması"):
        st.write(""" Örnek Sistem Açıklaması """)
      

     
 
    
if __name__ == '__main__':
    main()

st.write("")
st.write("")
st.write("TİTANİC VERİ SETİ")
st.write("")

Sex_dict = {'female':0,'male':1}
Embarked_dict = {'C':0,'S':1,'Q':2}


PClass=st.number_input("Bilet sınıfı",1,3)
Sex=st.selectbox("Sex/Cinsiyet", tuple(Sex_dict.keys()))
Age=st.number_input("Age/Yaş",0,100)
SibSp=st.number_input("Titanic'deki kardeşsayısı", 0,10)
Parch=st.number_input("Ebeveyn çocuk sayısı",0,10)
Fare=st.number_input("Ücret",0.00,500.00)
Embarked=st.selectbox("Biniş limanı",tuple(Embarked_dict.keys()))



Sex=Sex_dict.get(Sex)
Embarked=Embarked_dict.get(Embarked)





res = pd.DataFrame(data =
        {'PClass':[PClass],'Sex':[Sex],'Age':[Age],
         'SibSp':[SibSp],'Parch':[Parch],'Fare':[Fare],
          'Embarked':[Embarked]
          })

import pickle
with open('RandomForestModel.pkl', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(res)
prediction = str(prediction).strip('[]')

if st.button('Tahmin/Predict'):
    st.write("Random Forest Modeli Tahmini/Random Forest Model Prediction: ",prediction)
    












    










    








    