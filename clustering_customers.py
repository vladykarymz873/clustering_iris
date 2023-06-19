import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

model = pickle.load(open('output_iriss.sav', 'rb'))

df = pd.read_excel("output_iris.xlsx")
features = ['Sepal_Length', 'Sepal_Width']
X = df[features]
st.title('Clustering Cre')

numClusters = st.slider("Select Number of Clusters", min_value=1, max_value=10, value=3)

model = KMeans(n_clusters=numClusters)
clusters = model.fit_predict(X)

fig, ax = plt.subplots()
scatter = ax.scatter(X['Sepal_Length'], X['Sepal_Width'], c=clusters, cmap='viridis')
ax.set_xlabel('Sepal_Length')
ax.set_ylabel('Sepal_Width')

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

st.pyplot(fig)
