import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import numpy as np

def main():
    st.set_page_config(page_title="Untersuchung der Parteien zur BTW 25 anhand des Wahl-O-Mat", page_icon="🗳️", layout="wide")
    st.subheader("🗳️ Untersuchung der Parteien zur BTW 25 anhand des Wahl-O-Mat")
    try:
        df=pd.read_csv(r"https://raw.githubusercontent.com/tobiarnold/Wahlomat25/refs/heads/main/df_percent.csv")
        df.set_index(df.columns[0], inplace=True)
    except:
         st.write("Daten können nicht geladen werden.")
    try:
            matrix = df.values
            linkage = sch.linkage(matrix, method="ward")  
            dendro = sch.dendrogram(linkage, labels=df.index, no_plot=True)
            sorted_indices = dendro["leaves"]  
            df_reordered = df.iloc[sorted_indices, sorted_indices]  
            fig = go.Figure(data=go.Heatmap(
                        z=df_reordered.values,
                        x=df_reordered.columns,  
                        y=df_reordered.index,  
                        colorscale="portland",
                        showscale=False,
                        text=df_reordered.values,
                        texttemplate="%{text}"
                    ))
            fig.update_layout(
                        title="Übereinstimmungen laut Wahl-O-Mat zwischen den Parteien in %",
                        xaxis=dict(
                            tickangle=-45
                        ),
                         dragmode='pan'
                    )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except:
         st.write("Heatmap kann nicht gealden werden")
    try:
        df_pca = df.drop("Verjüngungsforschung", axis=0) 
        df_pca = df_pca.drop("Verjüngungsforschung", axis=1)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_pca)
        pca_df = pd.DataFrame(pca_result, index=df_pca.index, columns=['PCA1', 'PCA2'])
        dist_matrix = pairwise_distances(pca_df)  
        similarity_scores = np.exp(-dist_matrix)  
        plt.figure()
        scatter = plt.scatter(pca_df["PCA1"], pca_df["PCA2"])  
        for i in pca_df.index:
            plt.annotate(i, (pca_df.loc[i, "PCA1"], pca_df.loc[i, "PCA2"]), fontsize=10)
        plt.title("Parteinähe basierend auf Ähnlichkeitswerten (ohne die Partei Verjüngungsforschung)")
        plt.xticks([])  
        plt.yticks([]) 
        plt.grid(False) 
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        st.pyplot(plt)
    except:
         st.write("Bild kann nicht dargestellt werden.")
if __name__ == "__main__":
  main()
