import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('kr-vs-kp.data')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
print(data.head(5))


#------------------------------------------------------------------------------------------------------------------------
#Separando conjunto de treinamento e de teste
X = data.iloc[:, :-1]  # Atributos (todas as colunas, exceto a última)
y = data.iloc[:, -1]   # Coluna de saída (última coluna)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print("Atributos de treinamento:")
#print(X_train)

#print("\nAtributos de teste:")
#print(X_test)


#tirar coluna
#print(data['saida'])   printar coluna
#print(data.coluna4)    printar coluna(nome sem espaço)

#print(data.iloc[1])     printar linha

#------------------------------------------------------------------------------------------------------------------------
# Obter os valores únicos na coluna 'Coluna'
df = pd.DataFrame(data)

# Verificar a contagem de ocorrências de cada valor único na coluna 'Coluna'
#for a in range(1, 37):
#    contagem_ocorrencias = df[('coluna'+str(a))].value_counts()
#    print(contagem_ocorrencias)



#------------------------------------------------------------------------------------------------------------------------
# Converter atributos não numéricos em numéricos
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Calcular a matriz de correlação
corr_matrix = data.corr()


# Calcular a matriz de correlação
corr_matrix = data.corr()

# Exibir a matriz de correlação
print(corr_matrix)

# Visualizar a matriz de correlação em um heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


#------------------------------------------------------------------------------------------------------------------------
