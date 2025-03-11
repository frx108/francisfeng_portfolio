#### francisfeng_portfolio

# The Metropolitan Museum of Art Exhibitions: Data Visualization

## Project Description
This project visualizes exhibition trends at The Metropolitan Museum of Art (The Met) from 2015 to 2024. Using data science techniques such as PCA analysis, topic modeling, clustering, and geospatial mapping, the project explores thematic patterns, global reach, and exhibition popularity at The Met.

## Installation Instructions
1. Clone the repository:
   ```sh
   git clone https://github.com/frx108/francisfeng_portfolio.git
   cd met_exhibitions
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Install specific packages if needed (e.g., folium):
   ```sh
   pip install folium
   ```
   Then, import it in your script:
   ```python
   import folium
   ```
4. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Dataset Source & Retrieval
- Data was scraped from The Met's official website using a Python web scraping script: https://www.metmuseum.org/exhibitions/past.
- See metexhibitions_scrape.ipynb for details on data collection, including exhibition titles, descriptions, dates, and URLs.

## Visualizations
1. **The MET Exhibitions by Theme (PCA Analysis)**
        - Uses Principal Component Analysis (PCA) and K-Means clustering to identify thematic groupings of exhibitions based on their descriptions.  
        - Shows how exhibitions cluster into themes such as "Contemporary Art," "Photography & Visual Media," and "Cultural Heritage." 

2. **Evolution of Curatorial Themes (2015-2024)**
        - Applies Non-Negative Matrix Factorization (NMF) topic modeling to analyze curatorial trends over time.  
        - Reveals shifts in focus, such as increased attention to contemporary artists or heritage conservation.

3. **Global Coverage of The MET Exhibitions**
        - A Folium-based interactive map visualizing the geographic coverage of exhibitions.
        - Highlights The Met's global curatorial scope and regional exhibition trends.

4. **Top 5 Most Visited Exhibitions (2015-2017)**
        - A treemap displaying the most attended exhibitions of 2015, 2016, and 2017.
        - Provides insight into visitor preferences and the impact of blockbuster exhibitions.

### Visualization Selection Rationale
    - Each visualization was chosen to highlight a specific aspect of The Met's exhibition data, balancing thematic, temporal, geographic, and audience-focused perspectives.  
    - PCA and clustering reveal latent themes among exhibitions, topic modeling tracks curatorial evolution, mapping shows spatial distribution, and attendance data captures visitor engagement. 

## Limitations & Outlook  

### 1. The MET Exhibitions by Theme (PCA Analysis)  

**Limitations:**

    - PCA reduces high-dimensional textual data into two principal components, which may oversimplify complex thematic relationships.  
    - K-Means clustering requires predefined cluster numbers, potentially overlooking nuanced exhibition themes.  
    - The analysis relies on exhibition descriptions, which may not fully capture thematic intent or curator perspectives.  

**Future Improvements:**

    - Experimenting with t-SNE or UMAP for a more nuanced representation of exhibition similarities.  
    - Using hierarchical clustering or DBSCAN to detect organic groupings without predefined clusters.  
    - Incorporating curator interviews or additional metadata (e.g., exhibition medium, artist demographics) for deeper insights.  

### 2. Evolution of Curatorial Themes (2015-2024)  

**Limitations:**  

    - NMF topic modeling depends on the quality and granularity of exhibition descriptions.  
    - Some themes may be artifacts of wording rather than genuine curatorial shifts.  
    - The approach does not account for external influences, such as museum funding or societal trends, on curatorial decisions.  

**Future Improvements:**  

    - Comparing results with Latent Dirichlet Allocation (LDA) or BERTopic to validate thematic structures.  
    - Integrating curator notes, press releases, or visitor feedback to contextualize trends.  
    - Expanding the time frame to assess whether trends persist or fluctuate over longer periods.  

### 3. Global Coverage of The MET Exhibitions  

**Limitations:**  

    - The map reflects exhibition origins but does not indicate the depth of engagement with each region.  
    - Certain exhibitions may be multidisciplinary or have multiple geographic influences, which the visualization does not capture.  
    - The approach does not differentiate between internally curated exhibitions and traveling exhibitions from external institutions.  

**Future Improvements:**  

    - Adding weighted markers or choropleth layers to indicate the frequency of exhibitions from each region.  
    - Cross-referencing exhibition data with loan agreements or collaboration details to better understand global partnerships.  
    - Implementing timeline-based animations to show shifts in geographic focus over time.  

### 4. Top 5 Most Visited Exhibitions (2015-2017)  

**Limitations:**  

    - Data is limited to a short time frame and does not reflect long-term visitor trends.  
    - External factors like ticket pricing, seasonality, or concurrent events may have influenced visitor numbers.  
    - The treemap format provides a snapshot but does not reveal visitor demographics or engagement levels.  

**Future Improvements:**  

    - Expanding the dataset to include more years and assess evolving visitor preferences.  
    - Incorporating survey data or social media analytics to understand visitor engagement beyond attendance numbers.  
    - Comparing exhibition popularity with marketing efforts to identify factors driving high attendance.  

## Contributions
Contributions are welcome! If you'd like to improve visualizations, enhance interactivity, or add new datasets, feel free to submit a pull request.

## Acknowledgments
This project leverages data visualization and machine learning techniques to analyze The Met's exhibitions, showcasing trends and patterns in curatorial practices.

---
*Author: Francis Feng*

