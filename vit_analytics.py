import streamlit as st
from slugify import slugify

from time import sleep
import datetime
import pandas as pd
import os
from io import StringIO
import time
import numpy as np
import json # The module we need to decode JSON
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode


st.set_page_config(
		page_title= "Vitreo Network Analytics", # String or None. Strings get appended with "• Streamlit".
		 layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
		 initial_sidebar_state="collapsed",  # Can be "auto", "expanded", "collapsed"
		 page_icon="Vitreo",  # String, anything supported by st.image, or None.
)

df_cellinfo = []
num_new_rows = 1000 #st.sidebar.number_input("Add Rows",1,50)
#ncol = st.session_state.df.shape[1]  # col count
#st.title("Network map")

col1, col2, col3 = st.columns([1.6, 3, 4])

col1.header("Vitreo")
#col1.subheader("Vitreo")
from os import listdir
all_files = listdir("./")   
csv_files = list(filter(lambda f: f.endswith('.csv'), all_files)) 


col2.subheader("Select")
col3.subheader("Location")

with col1:
    #st.write('You selected:', uploaded_file)
    fig1place = st.empty()

    time.sleep(1)
    selected_file = st.selectbox('Select sensor records file',(csv_files))

    st.write('Selected file:', selected_file)


    start_date = st.date_input(
        "Start date",
        datetime.date(2022, 5, 1))
    end_date = st.date_input(
        "End date",
        datetime.date.today())
with col2:
    selectiontablecontainer=st.empty()
    uoloadplace=st.empty()
    with uoloadplace.container():
        uploaded_file = st.file_uploader("Upload a sensor record file")
        #Saving upload
        if uploaded_file is not None:
            with open(os.path.join("./",uploaded_file.name),"wb") as f:
                f.write((uploaded_file).getbuffer())
        
            st.success("File Saved")
            st.experimental_rerun()




rw =1
#st.sidebar.header("Design Campaign")
now = datetime.datetime.now()
# dd/mm/YY H:M:S
cdate = now.strftime("%d/%m/%Y %H:%M:%S")

@st.cache(suppress_st_warning=True)
def generateDiscreteColourScale(colour_set):
    #colour set is a list of lists
    colour_output = []
    num_colours = len(colour_set)
    divisions = 1./num_colours
    c_index = 0.
    # Loop over the colour set
    for cset in colour_set:
        num_subs = len(cset)
        sub_divisions = divisions/num_subs
        # Loop over the sub colours in this set
        for subcset in cset:
            colour_output.append((c_index,subcset))
            colour_output.append((c_index + sub_divisions-
                .001,subcset))
            c_index = c_index + sub_divisions
    colour_output[-1]=(1,colour_output[-1][1])
    return colour_output

# Harry Potter Houses
#color_schemes = [
#    ['#890000','#890000','#5c0000'],
#    ['#2a6b28','#0b4c07','#003206'],
#    ['#4f5a90','#374798','#30375a']
#    ['#fff4b1','#ffed86','#ffdb00']
#]

color_schemes = [
    ['black'],
    ['green'],
    ['blue'],
    ['red']
]

colorscale = generateDiscreteColourScale(color_schemes)

import requests
import numpy as np


@st.cache

def convert_df(df):

    # IMPORTANT: Cache the conversion to prevent computation on every rerun

    return df.to_csv().encode('utf-8')


@st.cache(suppress_st_warning=True)
def get_geolocation(cellInfo):
    #The requests library makes it easy to call URLs using Python
    # APIkey='YOUR GOOGLE LOCATION API ENABLED KEY' #Insert your API key if you already have one & want to use it
    APIkey='AIzaSyDSWscmyq9hAFTzItxS4wZAxTak_mrIw4s'

    #print("input: ", cellInfo)
    
    if np.any(cellInfo['cellInfo_mcc']):

        mcc, mnc, tac, cell_id = cellInfo['cellInfo_mcc'], cellInfo['cellInfo_mnc'], cellInfo['cellInfo_tac'], cellInfo['cellInfo_cell_id']
        
        #Add your cell tower details here.
        postjson = {"cellTowers": [{"cellId": cell_id, "locationAreaCode": tac, 
                                    "mobileCountryCode": mcc, "mobileNetworkCode": mnc}]}

        #print(postjson)
        #Set the url to the appropriate API endpoint location
        url=None
        lat={}
        lon={}

        if APIkey:
            url='https://www.googleapis.com/geolocation/v1/geolocate?key={}'.format(APIkey)
        elif APIkey is not None:
            url="https://radiocells.org/backend/geolocate"
            
        if url:
            #Make the request
            r = requests.post(url, json=postjson)
        if not r.ok:
            # display the response if something went wrong...
            print('Error: '+r.text)
                

        #If we get a valid response
        if APIkey is not None and r.ok:
            #Obtain the JSON response to a Python dict object
            jsondata=r.json()

        lat = jsondata['location']['lat']
        lon = jsondata['location']['lng']
        accuracy = jsondata['accuracy']


    else:

        lat = np.nan
        lon = np.nan
        accuracy = np.nan

    #print('Output', cellInfo, ': ',lat, lon, accuracy)
    print("output: ", lat, lon, accuracy)
    return lat, lon, accuracy



fname = selected_file
usersDf = pd.read_csv(selected_file,  sep=','  , engine='python')

pd.set_option('display.max_columns', None)

usersDf['time'] = pd.to_datetime(usersDf['time']).dt.date

#usersDf.head()

mask = (usersDf['time'] >= start_date) & (usersDf['time'] <= end_date)
users_df_crp = usersDf.loc[mask]
#users_df_crp.head()
#st.write(users_df_crp)

#users_df_crp.sort_values(by='time', ascending=True, inplace=True)
data_org = pd.json_normalize(users_df_crp.data.apply(json.loads))
data_org.columns=data_org.columns.str.replace('.','_')
#st.write(data_org)

data_org.sort_values(by='ts', ascending=True, inplace=True)

data_org['readtime'] = pd.to_datetime(data_org['ts'], unit='s')
column_to_move = data_org.pop("readtime")
data_org.insert(0, "readtime", column_to_move)

data_org=data_org.fillna(0.00)


#data_org['ts']=pd.to_datetime(data_org['ts'])
#data_org['lon'] = data_org['lon'].astype(float)

for col in data_org.columns:
    try:
        data_org[col] = data_org[col].astype(float)
    except:
        data_org[col] = data_org[col].astype(str)

#AgGrid(data_org)
#st.write(data_org.shape)

dfno_net = data_org[data_org.cellInfo_cell_id == 0]

#AgGrid(dfno_net)
#st.write(dfno_net.shape)    

#df_unique_cell = (data_org.assign(cellInfo_cell_id = data_org.cellInfo_cell_id.where(data_org.cellInfo_cell_id!=0, 0))
#     .groupby('cellInfo_cell_id', as_index=False).tail(1))

#st.write(df_unique_cell)

df_unique_cell=data_org.groupby('cellInfo_cell_id').tail(1)
df_unique_cell = df_unique_cell[df_unique_cell.cellInfo_cell_id != 0]
#st.write(df_unique_cell)
#st.write(df_unique_cell.shape)   
frames = [df_unique_cell, dfno_net]

df_unique_cell_with_err = pd.concat(frames)

df_unique_cell_with_err.sort_values(by='ts', ascending=True, inplace=True)

#df_unique_cell_with_err['readtime'] = pd.to_datetime(df_unique_cell_with_err['ts'], unit='s')
#column_to_move = df_unique_cell_with_err.pop("readtime")

#df_unique_cell_with_err.insert(0, "readtime", column_to_move)

df_unique_cell_with_err.loc[df_unique_cell_with_err['cellInfo_mnc'] ==1.0, 'provider'] = 'Telekom'
df_unique_cell_with_err.loc[df_unique_cell_with_err['cellInfo_mnc'] ==2.0, 'provider'] = 'Vodafone'
df_unique_cell_with_err.loc[df_unique_cell_with_err['cellInfo_mnc'] ==3.0, 'provider'] = 'O2-ePlus'
df_unique_cell_with_err.loc[df_unique_cell_with_err['cellInfo_mnc'] ==0.0, 'provider'] = 'No Net'



df_dbg=[None]
df_dbg = df_unique_cell_with_err.filter(regex='dbg_')
#df_dbg['readtime'] = df_unique_cell_with_err['readtime'].copy()
#df_dbg.head()
#df_dbg.shape

df_cellinfo = [None]
df_cellinfo = df_unique_cell_with_err.filter(regex='cellInfo_')
#df_cellinfo['readtime'] = df_unique_cell_with_err['readtime'].copy()

#df_cellinfo.head()


#df_cellinfo=df_cellinfo.head(5)    
dfObj = [None]
location_list = None
location_list=list(map(lambda s: get_geolocation(s),df_cellinfo.to_dict('records')))
dfObj= pd.DataFrame(location_list,columns = ['lat' , 'lon', 'acc'])

df_unique_cell_with_err['lat'] = dfObj['lat'].values
df_unique_cell_with_err['lon'] = dfObj['lon'].values
df_unique_cell_with_err['acc'] = dfObj['acc'].values

df_unique_cell_with_err['lat'] = df_unique_cell_with_err['lat'].astype(float)
df_unique_cell_with_err['lon'] = df_unique_cell_with_err['lon'].astype(float)
df_unique_cell_with_err['acc'] = df_unique_cell_with_err['acc'].astype(float)

df_unique_cell_with_err['provider']  = df_unique_cell_with_err['provider'].astype('string')
df_unique_cell_with_err['reset_reason']  = df_unique_cell_with_err['reset_reason'].astype('string')
df_unique_cell_with_err['cellInfo_mode']  = df_unique_cell_with_err['cellInfo_mode'].astype('string')


#df_unique_cell_with_err.head()

#AgGrid(df_unique_cell_with_err)
#st.write(df_unique_cell_with_err.shape)   

fname_locs= 'locs_'+fname

# if not df_unique_cell_with_err.empty:
#     df_unique_cell_with_err.to_csv(fname_locs)  
#     print("Locations saved")
# else:
#     df_unique_cell_with_err = pd.read_csv(fname_locs)
#     print("Saved locations loaded")


diff_minutes=((df_unique_cell_with_err['ts'].diff())/60).round(decimals = 1)
diff_hrs=((df_unique_cell_with_err['ts'].diff())/3600).round(decimals = 1)

df_unique_cell_with_err.insert(1, "dt_hrs", diff_hrs)
df_unique_cell_with_err.insert(2, "dt_minutes", diff_minutes)

df_unique_cell_with_err.info()
    #The plot
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_unique_cell_with_err['readtime'], y=df_unique_cell_with_err['dt_minutes'],
                mode="markers+lines",
                name='Send Interval',
                #showlegend = True,
                #hovermode="x unified",
                customdata=df_unique_cell_with_err,
                text=df_unique_cell_with_err['cellInfo_mode'],
                hovertext=df_unique_cell_with_err['cellInfo_band'],
                #hoverlabel=dict(namelength=0),
                textposition = "bottom right",
                hovertemplate="<br>".join([
                    "Send interval (minutes): %{customdata[2]}",
                    "Send interval (hours): %{customdata[1]}",
                    "Network: %{customdata[40]}",
                    "Band: %{hovertext}",
                    "Mode: %{text}",
                    "Cell_ID: %{customdata[24]}",
                    "Date: %{customdata[0]}"])
                    ))

# Edit the layout
fig1.update_layout(
                xaxis_title='Date',
                yaxis_title='Interval / min')

#fig.update_traces(mode="markers+lines", hovertemplate=None)
#fig.update_layout(hovermode="x unified")

#fig.update_traces(line_color = "maroon")

#st.line_chart(df_unique_cell_with_err['readtime']['dt_minutes'])


#AgGrid(df_unique_cell_with_err)


gps_df=df_unique_cell_with_err.copy()

gps_df['lat'] = gps_df['lat'].interpolate(method='linear',limit=10, limit_direction='both')
gps_df['lon'] = gps_df['lon'].interpolate(method='linear',limit=10, limit_direction='both')
gps_df['acc'] = gps_df['acc'].interpolate(method='linear',limit=10, limit_direction='both')

#AgGrid(gps_df)
#st.write(gps_df.shape)  


api_token = "pk.eyJ1IjoicXVhbnR1bTEwIiwiYSI6ImNsMzdnaHVpejBvdWkzZG51dGF0ZHgwd24ifQ.Q9wJGGwf1Bt7cePTCRh-tw" # you need your own token
gps_df.info()

import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle

with col2:
    with selectiontablecontainer.container():
        gb = GridOptionsBuilder.from_dataframe(gps_df)
        gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
        gb.configure_side_bar() #Add a sidebar
        gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
        gridOptions = gb.build()

        grid_response = AgGrid(
            gps_df,
            gridOptions=gridOptions,
            data_return_mode='AS_INPUT', 
            update_mode='MODEL_CHANGED', 
            fit_columns_on_grid_load=False,
            theme='blue', #Add theme color to the table
            enable_enterprise_modules=True,
            height=680, 
            width='100%',
            reload_data=True
        )

        selected = grid_response['selected_rows']

#AgGrid(gps_df)
with col3:
    if np.any(selected):
        gps_df = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        mode = "markers+text",
        name = "Details",
        lon = list(gps_df['lon']),
        lat =list(gps_df['lat']),
        #marker_color=next(palette),
        marker=dict(
                size=list(gps_df['acc']/100),
                color= gps_df['cellInfo_mnc'],
                symbol="circle",
                opacity=0.7,
                showscale=False,
                colorscale=colorscale,  
                cmin=0,
                cmax=3
            ),
        #line={'color': 'red'},
        
        #marker = {'size': list(gps_df['acc']/100), 'color': gps_df['cellInfo_mnc'], 'symbol':"circle", 'opacity':0.7},
        
        customdata=gps_df,
        text=gps_df['cellInfo_mode'],
        hovertext=gps_df['cellInfo_band'],
        #hoverlabel=dict(namelength=0),
        textposition = "bottom right",
        hovertemplate="<br>".join([
                    "Network: %{customdata[40]}",
                    "Band: %{hovertext}",
                    "Range: %{customdata[43]} m",
                    "Mode: %{text}",
                    "Cell_ID: %{customdata[24]}",
                    "Date: %{customdata[0]}"])
        
        
        ))

    fig.update_layout(font_size=10,  title={'xanchor': 'center','yanchor': 'top', 'y':1, 'x':1,}, 
            title_font_size = 24, mapbox_accesstoken=api_token, width=700*1.5, height=450*1.5)

    fig.update_layout(
        margin ={'l':5,'t':5,'b':5,'r':5},
        mapbox = {
            'center': {'lon':gps_df['lon'].mean(), 'lat': gps_df['lat'].mean()},
            'zoom': 6},showlegend = False)
        
    st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': False,
            'editable': False,
        })



fig1.update_layout(font_size=10,  title={'xanchor': 'center','yanchor': 'top', 'y':1, 'x':1,}, 
            title_font_size = 24, mapbox_accesstoken=api_token, width=700*1.5, height=150*1.5)
fig1.update_layout(
    margin ={'l':5,'t':5,'b':5,'r':5},
    mapbox = {
        'center': {'lon':gps_df['lon'].mean(), 'lat': gps_df['lat'].mean()},
        'zoom': 7},showlegend = False)
with col1:
    with fig1place.container():
        st.write("**Sending intervals**")
        st.plotly_chart(fig1, use_container_width=True, config={
            'displayModeBar': False,
            'editable': False,
        })

csv = convert_df(df_unique_cell)
with col1:
    st.download_button( 

    label="Download data as CSV",

    data=csv,

    file_name=fname_locs,

    mime='text/csv',

    )

with col3:
    
    st.write("**Sending intervals**")
    st.plotly_chart(fig1, use_container_width=True, config={
        'displayModeBar': False,
        'editable': False,
    })