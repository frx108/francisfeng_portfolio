
### STEP 1: INSTALL AND IMPORT PACKAGES ###

# Install packages
# pip install streamlit
# pip install plotly
# pip install folium

# Import packages
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import folium
import csv
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import spacy
import random


### STEP 2: CREATE FUNCTIONS FOR VISUALIZATIONS ###

# PCA Visualization
cluster_names = {
    0: "Cultural Heritage", 1: "Christmas Tree", 2: "Jefferson R. Burdick Collection", 3: "Contemporary Art", 
    4: "P.S. Art Annual Exhibition", 5: "Textural & Sculptural Art", 6: "Sport & Game", 7: "Drawings and Prints Department Collection", 
    8: "The Met Collection", 9: "Photography, Paintings & Other Visual Media"
}

def perform_pca():
    df = pd.read_csv("metexhibitions_2015-2024_pos.csv")

    df = df.dropna(subset=["Description_filtered", "Title"])
    documents = df["Description_filtered"].tolist()
    titles = df["Title"].tolist()
    years = df["Year"].astype(str).tolist()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(documents)

    tfidf_dense = tfidf_matrix.toarray()

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tfidf_dense)

    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_result)

    pca_df = pd.DataFrame({
        "PCA1": pca_result[:, 0],
        "PCA2": pca_result[:, 1],
        "Cluster": cluster_labels,
        "Year": years,
        "Title": titles
    })

    pca_df['Cluster Name'] = pca_df['Cluster'].map(cluster_names)

    fig = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        color="Cluster Name",
        hover_data={"Title": True, "Year": True},
        labels={"PCA1": "PCA Component 1", "PCA2": "PCA Component 2"}
    )
    
    fig.update_layout(
        height=600,
        legend_font=dict(size=18)
    )
    
    fig.update_traces(marker=dict(size=6)) 
    
    return pca_df, fig


# PCA Cluster Zoomed-In Visualization
def zoom_to_cluster(pca_df, cluster_number):
    
    cluster_data = pca_df[pca_df["Cluster"] == cluster_number]
    x_min, x_max = cluster_data["PCA1"].min(), cluster_data["PCA1"].max()
    y_min, y_max = cluster_data["PCA2"].min(), cluster_data["PCA2"].max()

    x_padding = (x_max - x_min) * 0.2
    y_padding = (y_max - y_min) * 0.2

    fig = px.scatter(
        cluster_data,
        x="PCA1",
        y="PCA2",
        color_discrete_sequence=["red"],
        hover_data={"Title": True, "Year": True},
        labels={"PCA1": "PCA Component 1", "PCA2": "PCA Component 2"}
    )

    fig.update_layout(
        width=1000,
        height=500, 
        margin=dict(l=50, r=50, t=50, b=50)
    )

    fig.update_xaxes(range=[x_min - x_padding, x_max + x_padding])
    fig.update_yaxes(range=[y_min - y_padding, y_max + y_padding])
    fig.update_traces(marker=dict(size=11)) 
    
    return fig


# Topic Modeling Visualization
def topic_model():
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv("metexhibitions_2015-2024_pos.csv")
    
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorized_data = vectorizer.fit_transform(df["Description_filtered"])
    
    nmf = NMF(n_components=10, random_state=1)
    doc_topic_dist_nmf = nmf.fit_transform(vectorized_data)
    
    topic_words_df = pd.DataFrame(nmf.components_, columns=vectorizer.get_feature_names_out())
    for topic, topic_row in topic_words_df.iterrows():
        top_10_words = ", ".join(topic_row.sort_values(ascending=False).head(10).index)
        print(f"Topic {topic}: {top_10_words}")
    
    topics = []
    for topic_num, row in topic_words_df.iterrows():
        top_5_words = row.sort_values(ascending=False).head(5).index.to_list()
        topic_label = ", ".join(top_5_words)
        topics.append(topic_label)
    
    df_topics = pd.DataFrame(doc_topic_dist_nmf, index=df.Year, columns=topics)
    df_topics_per_year = df_topics.groupby("Year").mean()
    
    fig = px.line(
        df_topics_per_year,
        markers=True,
        labels={"value": "Topic Weight", "variable": "Topic"}
    )
    
    fig.update_layout(
        height=600,
        legend_font=dict(size=18)
    )
    
    fig.update_traces(marker=dict(size=8)) 
    
    return fig


# Folium Map Visualization
def create_world_map(csv_file):
    m = folium.Map(location=[20, 0], zoom_start=2)

    with open(csv_file, encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            title, location, lat, lon = row
            lat, lon = float(lat), float(lon)
            
            jitter_scale = 0.1
            lat += random.uniform(-jitter_scale, jitter_scale)
            lon += random.uniform(-jitter_scale, jitter_scale)        

            custom_icon = folium.CustomIcon(icon_image="museum_icon3.png", icon_size=(35, 50))

            folium.Marker(
                location=[lat, lon],
                popup=f"<b>{title}</b><br>{location}",
                tooltip=title,
                icon=custom_icon
            ).add_to(m)
    
    return m


# Tree Map Visualization
def treemap():
    df = pd.read_csv("popularexhibitions.csv")
    df["Year"] = df["Year"].astype(str)

    fig = px.treemap(df,
                     path=['Year', 'Title'],  # Hierarchy: Year → Exhibition Title
                     values='Visitor',
                     color='Visitor',  # Color based on visitor count for gradient effect
                     color_continuous_scale='Blues'  # Change to another scale if preferred
                    )
    
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Visitors: %{value}<extra></extra>"
    )
    
    
    fig.update_layout(
        width=800,
        height=800, 
        font=dict(size=40),
        coloraxis_colorbar=dict(title="Visitors")  # Adds a color legend
    )    
    
    return fig



### STEP 3: CREATE STREAMLIT LAYOUT ###

# Basic Layout & Heading
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Exhibitions at the Metropolitan Museum of Art</h1>", unsafe_allow_html=True)

st.subheader(
    '''This project visualizes exhibition trends at The Metropolitan Museum of Art (The Met) from 2015 to 2024. Using data science techniques such as PCA analysis, topic modeling, clustering, and geospatial mapping, the project explores thematic patterns, global reach, and exhibition popularity at The Met. The data come from the Met's official website: https://www.metmuseum.org/exhibitions/past.'''
)

st.divider()

# Call PCA Visualization Function
pca_df, pca_fig = perform_pca()

# Create column 1 and 2
col1, col2= st.columns([0.6, 0.4])

# Column 1: The MET Exhibitions by Theme (PCA Analysis)
with col1:
    st.subheader("The MET Exhibitions by Theme (PCA Analysis)")
    st.plotly_chart(pca_fig, use_container_width=True)

# Column 2: Zoomed-In View of The MET Exhibitions by Theme
with col2:
    st.subheader(f"Zoomed-In View of The MET Exhibitions by Theme")

    # Create a list of cluster names for the selectbox
    cluster_name_list = list(cluster_names.values())
    
    # Dropdown for clusters
    cluster_selection_name = st.selectbox(
        "Select a Theme to Zoom In",
        options=cluster_name_list,
        index=0
    )
    
    # Map the selected name back to the numeric cluster value
    cluster_selection = [k for k, v in cluster_names.items() if v == cluster_selection_name][0]
    # Call Cluster Zoomed-In Visualization Function
    zoomed_in_fig = zoom_to_cluster(pca_df, cluster_selection)
    st.plotly_chart(zoomed_in_fig, use_container_width=True)

st.markdown('''
            1. **The MET Exhibitions by Theme (PCA Analysis)**  
   - Uses Principal Component Analysis (PCA) and K-Means clustering to identify thematic groupings of exhibitions based on their descriptions.  
   - Shows how exhibitions cluster into themes such as "Contemporary Art," "Photography & Visual Media," and "Cultural Heritage." 
   ''')

st.divider()

# Create column 3
col3 = st.columns([1.0])

# Column 3: Evolution of Curatorial Themes at The MET (2015-2024)
with col3[0]:
    st.subheader("Evolution of Curatorial Themes at The MET (2015-2024)")
    # Call Topic Modeling Visualization Function
    line_fig = topic_model()
    st.plotly_chart(line_fig, use_container_width=True)

st.markdown('''
            2. **Evolution of Curatorial Themes (2015-2024)**  
   - Applies Non-Negative Matrix Factorization (NMF) topic modeling to analyze curatorial trends over time.  
   - Reveals shifts in focus, such as increased attention to contemporary artists or heritage conservation. 
   ''')

st.divider()

# Create column 4
col4 = st.columns([0.5])

# Column 4: Global Coverage of The MET Exhibitions
with col4[0]:
    st.subheader("Global Coverage of The MET Exhibitions")
    # Call Folium Map Visualization Function
    world_map = create_world_map("exhibition_locations.csv")
    folium_static(world_map, width=1800, height=600)
    
st.markdown('''
            3. **Global Coverage of The MET Exhibitions**  
   - A Folium-based interactive map visualizing the geographic coverage of exhibitions.
   - Highlights The Met's global curatorial scope and regional exhibition trends.
   ''')

st.divider()
 
# Create column 5
col5 = st.columns([1.0])
   
# Column 5: Top 5 Most Visited The MET Exhibitions (2015-2017)
with col5[0]:
    st.subheader("Top 5 Most Visited The MET Exhibitions (2015-2017)")
    # Call Tree Map Visualization Function
    treemap_fig = treemap()
    st.plotly_chart(treemap_fig, use_container_width=True)

st.markdown('''
            4. **Top 5 Most Visited Exhibitions (2015-2017)**  
   - A treemap displaying the most attended exhibitions of 2015, 2016, and 2017.
   - Provides insight into visitor preferences and the impact of blockbuster exhibitions.
   ''')

st.divider()