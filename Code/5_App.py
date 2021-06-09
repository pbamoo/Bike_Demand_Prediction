#!/usr/bin/env python
# coding: utf-8

# ### Building Simple Web Application

# In[1]:


#import necessary libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Import dataset
df = pd.read_csv('../data/train.csv', header = 0, error_bad_lines=False)
df = df.set_index('datetime')


# In[3]:


#create the app instance
app = dash.Dash(__name__)


# In[4]:


#import an external CSS file : for our app to look a bit nicer
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


# ### 4. Load the trained objects

# In[5]:


model = joblib.load('bike-model.joblib')
pca = joblib.load('pca.joblib')
scaler = joblib.load('scaler.joblib')


# ### 5. Build the input components and their respective divs
#  We will use input boxes to input the values of numerical features and drop-down menus for the categorical ones. We will place each of the nine inputs (6 numerical & 3 categorical) in its own four-column div.

# In[6]:


numerical_features = ['month', 'hour', 'temp', 'atemp', 'humidity','windspeed', 'count']
options_dropdown = [{'label':x.upper(), 'value':x} for x in numerical_features]

dd_x_var = dcc.Dropdown(
        id='x-var',
        options = options_dropdown,
        value = 'temp'
        )

div_x_var = html.Div(
        children=[html.H4('Variable for x axis: '), dd_x_var],
        className="six columns"
        )

dd_y_var = dcc.Dropdown(
        id='y-var',
        options = options_dropdown,
        value = 'count'
        ) 

div_y_var = html.Div(
        children=[html.H4('Variable for y axis: '), dd_y_var],
        className="six columns"
        )


# In[7]:


### 1. For numerical features

## Div for month
input_month = dcc.Input(
    id='month',
    type='numeric',
    value=1)

div_month = html.Div(
        children=[html.H3('month:'), input_month],
        className="four columns"
        )

## Div for hour
input_hour = dcc.Input(
    id='hour',
    placeholder='',
    type='numeric',
    value=4)

div_hour = html.Div(
        children=[html.H3('hour:'), input_hour],
        className="four columns"
        )

## Div for temp
input_temp = dcc.Input(
    id='temp',
    type='numeric',
    value=9.84)

div_temp = html.Div(
        children=[html.H3('temp:'), input_temp],
        className="four columns"
        )

## Div for atemp
input_atemp = dcc.Input(
    id='atemp',
    placeholder='',
    type='numeric',
    value=14.40)

div_atemp = html.Div(
        children=[html.H3('atemp:'), input_atemp],
        className="four columns"
        )

## Div for humidity
input_humidity = dcc.Input(
    id='humidity', 
    placeholder='',
    type='numeric',
    value=81)

div_humidity = html.Div(
        children=[html.H3('humidity:'), input_humidity],
        className="four columns"
        )

## Div for windspeed
input_windspeed = dcc.Input(
    id='windspeed', 
    placeholder='',
    type='numeric',
    value=0.0)

div_windspeed = html.Div(
        children=[html.H3('windspeed: '), input_windspeed],
        className="four columns"
        )


# In[8]:


### 2. For categorical features

## Div for day_typ
day_typ_values = ['weekend', 'holiday', 'workday']
day_typ_options = [{'label': x, 'value': x} for x in day_typ_values]
input_day_typ = dcc.Dropdown(
    id='day_typ',
    options = day_typ_options,
    value = 'workday'
    )

div_day_typ = html.Div(
        children=[html.H3('day_typ:'), input_day_typ],
        className="four columns"
        )

## Div for season
season_values = ['1', '2', '3', '4']
season_options = [{'label': x, 'value': x} for x in season_values]
input_season = dcc.Dropdown(
    id='season', 
    options = season_options,
    value = '1'
    )

div_season = html.Div(
        children=[html.H3('season:'), input_season],
        className="four columns"
        )

## Div for weather
weather_values = ['1', '2', '3', '4']
weather_options = [{'label': x, 'value': x} for x in weather_values]
input_weather = dcc.Dropdown(
    id='weather', 
    options = weather_options,
    value = '1'
    )

div_weather = html.Div(
        children=[html.H3('weather:'), input_weather],
        className="four columns"
        )


# In[9]:


#### Further group the nine inputs into two sub-groups & use a div for each group:
## Div for numerical characteristics
div_numerical = html.Div(
        children = [div_month, div_hour, div_temp, div_atemp, div_humidity, div_windspeed],
       className="row"
        )

## Div for categorical features
div_categorical = html.Div(
        children = [div_day_typ, div_season, div_weather],
        className="row"
        )


# ### 6. Build the prediction function

# In[10]:


def get_prediction(month, hour, temp, atemp, humidity, windspeed, day_typ, season, weather):
    '''takes the inputs from the user and produces the count prediction'''
    
    cols = ['month', 'hour', 'humidity', 'windspeed', 'season_1', 'season_2', 'season_3', 'season_4', 'day_typ_holiday', 
            'day_typ_weekend', 'day_typ_workday', 'weather_1', 'weather_2', 'weather_3', 'weather_4','mtemp']

    day_typ_dict = {x: 'day_typ_' + x for x in day_typ_values[1:]}
    season_dict = {x: 'season_' + x for x in season_values[1:]}
    weather_dict = {x: 'weather_' + x for x in weather_values[1:]}
    
    ## produce a dataframe with a single row of zeros
    df = pd.DataFrame(data = np.zeros((1,len(cols))), columns = cols)
    
    ## get the numeric characteristics
    df.loc[0,'month'] = month
    df.loc[0,'hour'] = hour
    df.loc[0,'temp'] = temp
    df.loc[0,'atemp'] = atemp
    df.loc[0,'humidity'] = humidity
    df.loc[0,'windspeed'] = windspeed
    
    ## transform dimensions into a single mtemp using PCA
    dims_df = pd.DataFrame(data=[[temp, atemp]], columns=['temp', 'atemp'])
    df.loc[0,'mtemp'] = pca.transform(dims_df).flatten()[0]
    
    ## Use the one-hot encoding for the categorical features
    if day_typ!='weekend':
        df.loc[0, day_typ_dict[day_typ]] = 1
    
    if season!='2':
        df.loc[0, season_dict[season]] = 1
    
    if weather != '2':
        df.loc[0, weather_dict[weather]] = 1
    
    ## Scale the numerical features using the trained scaler
    numerical_features = ['humidity', 'windspeed', 'mtemp']
    df.loc[:,numerical_features] = scaler.transform(df.loc[:,numerical_features])
    
    ## Get the predictions using our trained neural network
    prediction = model.predict(df.values).flatten()[0]
    
    ## Transform the log-counts to counts
    prediction = np.exp(prediction)
   
    return int(prediction)
    


# ### 7. Create the layout of the application

# In[11]:


# for histogram
trace = go.Histogram(
    x = df['count']
    )

layout = go.Layout(
    title = 'Bike Demand Distribution',
    xaxis = dict(title='Demand'),
    yaxis = dict(title='Count')
    )

figure = go.Figure(
    data = [trace],
    layout = layout
    )


# In[12]:


## App layout
app.layout = html.Div([
        html.H1('IDR Predict bike counts'),
        
        html.H2('1. Enter the bike characteristics to get the predicted count'),
        
        html.Div(
                children=[div_numerical, div_categorical]
                ),
        html.H1(id='output',
                style={'margin-top': '50px', 'text-align': 'center'}),
        html.H2('2. Interactive scatter plot of the numerical features'),
        html.P('Select your x and y features to view plot'),
        html.Div(
                children=[div_x_var, div_y_var],
                className="row"
                ), 
        dcc.Graph(id='scatter'),
        html.H2('3. Distribution of bike counts'),
        html.P('This is the original distribution of the bike counts.'), 
        dcc.Graph(id='histogram', figure=figure)
        ])


# ### 8.  Build the decorator (callback)

# In[13]:


#for predictor
predictors = ['month', 'hour', 'temp', 'atemp', 'humidity', 'windspeed', 'day_typ', 'season', 'weather']
@app.callback(
        Output('output', 'children'),
        [Input(x, 'value') for x in predictors])

def show_prediction(month, hour, temp, atemp, humidity, windspeed, day_typ, season, weather): 
    pred = get_prediction(month, hour, temp, atemp, humidity, windspeed, day_typ, season, weather)
    return str("Predicted count: {:,}".format(pred))


# In[14]:


# for scatter plot
@app.callback(
        Output(component_id='scatter', component_property='figure'),
        [Input(component_id='x-var', component_property='value'), Input(component_id='y-var', component_property='value')])


def scatter_plot(x_col, y_col):
    trace = go.Scatter(
            x = df[x_col],
            y = df[y_col],
            mode = 'markers'
            )
    
    layout = go.Layout(
            title = 'Scatter plot',
            xaxis = dict(title = x_col.upper()),
            yaxis = dict(title = y_col.upper())
            )
    
    output_plot = go.Figure(
            data = [trace],
            layout = layout
            )
    
    return output_plot


# ### 9. Code to run the server

# In[15]:


if __name__ == '__main__':
    app.run_server(debug=False) #setting debug as false instead of true works


# The aim was to build a basic web app a user can input the details and the model will predict the number of bikes that will be in demand at a particular time, the app is semi operational, some minor tweaks are needed to get the prediction aspect going as well as modify the layout to have different pages with different info.

# In[ ]:





# In[ ]:




