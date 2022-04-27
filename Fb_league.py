import streamlit as st
import pandas as pd
import numpy as np
import base64
import seaborn as sns
import matplotlib.pyplot as plt


st.title('National Football League Stats (Rushing) Explorer')

st.markdown("""
This app performs simple webscraping of NFL Football player stats data (focusing on Rushing)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [pro-football-reference.com](https://www.pro-football-reference.com/).
""")

st.sidebar.header('USER INPUT')
selected_year = st.sidebar.selectbox('YEAR', list(reversed(range(1990,2022))))

# Web scraping of NFL player stats
# https://www.pro-football-reference.com/years/2022/rushing.htm
@st.cache
def load_data(year):
    url = "https://www.pro-football-reference.com/years/" + str(year) + "/rushing.htm"
    html = pd.read_html(url, header = 1)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats
playerstats = load_data(selected_year)

# Sidebar - Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('TEAM', sorted_unique_team, sorted_unique_team)

# Sidebar - Position selection
unique_pos = ['RB','QB','WR','FB','TE','DT','DB']
selected_pos = st.sidebar.multiselect('POSITION', unique_pos, unique_pos)

# Filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

# Download NFL player stats data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
       fig, ax = plt.subplots(figsize=(7, 5))
       ax = sns.heatmap(corr, mask=mask, vmax=1,cmap="YlGnBu",linewidths=.5,square=True)
       st.pyplot(fig)


#Clustermap     
if st.button('Intercorrelation Clustermap'):
    st.header('Intercorrelation Matrix Clustermap')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    with sns.axes_style("white"):
       fig, ax = plt.subplots(figsize=(7, 5))
       g = sns.clustermap(corr,cmap="vlag",vmin=0,vmax=10)
       st.pyplot(g)

       st.set_option('deprecation.showPyplotGlobalUse', False)

