# Importando as bibliotecas necessárias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregando o conjunto de dados
vinhos = datasets.load_wine()
P = vinhos.data
q = vinhos.target

# Dividindo o conjunto de dados em treino e teste
P_treina, P_testa, q_treina, q_testa = train_test_split(P, q, test_size=0.2, random_state=42)

# Inicializando o classificador k-NN (k-Nearest Neighbors)
classificador_knn = KNeighborsClassifier(n_neighbors=3)

# Treinando o classificador com os dados de treino
classificador_knn.fit(P_treina, q_treina)

# Fazendo previsões com os dados de teste
q_pred = classificador_knn.predict(P_testa)

# Avaliando a precisão do modelo
accuracy = accuracy_score(q_testa, q_pred)

print(f'A precisão do modelo é: {accuracy* 100:.2f}%')
# print(f'O conteúdo do dataset é: ')
# print(vinhos.DESCR) para exibir o conteúdo do dataset