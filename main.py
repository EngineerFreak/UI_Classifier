##################################################################################
# Project: Interface de usuário para classificadores
# Author: Rafael Alves (EngineerFreak)
# Created: Rafael Alves (EngineerFreak) - 14.02.2021
# Edited :
##################################################################################

import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

#importar os classificadores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

########## Functions ##########

# definir os datasets desejados para a UI
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Cancer de mama":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x, y

# para parametrizar os diferentes tipos de classificadores conferir a documentacao do scikit-learn.org
# e preparar os dados importantes para o classificador para a interface gráfica
def add_parameter_ui(classifer_name):
    params = dict()
    if classifer_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif classifer_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else: # random forest
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

# Definir os classificadores
def get_classifier(classifier_name, params):
    if classifier_name == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name == "SVM":
        classifier = SVC(C=params["C"])
    else: # random forest
        classifier = RandomForestClassifier(n_estimators=params["n_estimators"],
                                            max_depth=params["max_depth"],
                                           random_state= 1234)
    return classifier

# Classificacao

########## End Functions ##########


# para rodar este programa colocar na linha de comando: streamlit run [endereco_do_programa]/main.py
if __name__ == '__main__':
    st.title("""
    Interface de usuário (Streamlit) 
    ### by Rafael Alves 
    #### Instagram: @iamanengineerfreak
    """)

    st.write("""
    # Explorando os diferentes tipos de classificadores
    Qual é o melhor?
    """)

    dataset_name = st.sidebar.selectbox("Selecione o  dataset", ("Iris", "Cancer de mama", "Vinho"))
    # st.sidebar.write(dataset_name)

    classifier_name = st.sidebar.selectbox("Selecione o classificador", ("KNN", "SVM", "Random Forest"))
    # st.sidebar.write(classifier_name)

    x, y = get_dataset(dataset_name)
    st.write("formato do dataset", x.shape)
    st.write("número de classes", len(np.unique(y)))

    params = add_parameter_ui(classifier_name)
    classifier = get_classifier(classifier_name, params)

    # Precesso de Classificacao
    x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.2, random_state=5678)

    classifier.fit(x_treino, y_treino)
    y_predict = classifier.predict(x_teste)

    accuracy = accuracy_score(y_teste, y_predict)
    st.write(f"classificador = {classifier_name}")
    st.write(f"accurácia = {accuracy}")

    # Plot do resultado
    pca = PCA(2) # 2D
    x_projected = pca.fit_transform(x)
    x1 = x_projected[:, 0]
    x2 = x_projected[:, 1]

    figure = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.colorbar()

    #plt.show()
    st.pyplot(figure)

    #TODO
    # 0) Visao: Criar uma plataforma para o curso onde se poderá testar os diferentes algoritmos aprendidos.
    # 1) Adcionar mais parametros para os classificadores
    # 2) adcionar mais classificadores
    # 3) preparar o algoritmo para entrada de dados externos
    # 4) aumentar a quantidade de dados para classificacao
    # 5) Mais possibilidades de visualizacao de indicadores
    # 6) criar sistema interno para outros tipos de algoritmos dentro desta plataforma
