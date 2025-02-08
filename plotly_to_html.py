import pandas as pd
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import numpy as np
try:
    df = pd.read_csv(r"https://raw.githubusercontent.com/tobiarnold/Wahlomat25/refs/heads/main/df_percent.csv")
    df.set_index(df.columns[0], inplace=True)
except:
    print("Daten können nicht geladen werden.")

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
        texttemplate="%{text}",
        hoverinfo="x+y+text"
    ))
fig.update_layout(
        height=800,  
        width=1200, 
        xaxis=dict(tickangle=-90, fixedrange=True),  
        yaxis=dict(fixedrange=True),  
        margin=dict(l=50, r=50, t=50, b=50), 
    )
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

html_template = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Heatmap Plot</title>
        <style>
            h1 {{
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <h1>Übereinstimmungen laut Wahl-O-Mat zwischen den Parteien</h1>
        {plot_html}
        <img src="https://raw.githubusercontent.com/tobiarnold/Wahlomat25/refs/heads/main/PCA.png" alt="PCA-Analyse">
    </body>
    </html>
    """

with open("index.html", "w", encoding="utf-8") as file:
        file.write(html_template)

print("Speicherung ist erfolgt als index.html.")
