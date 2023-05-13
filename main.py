
import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import streamlit_vega_lite as st_vl
import random
import plotly.express as px
import joblib
#pd.set_option('max_columns',200)
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from xgboost import XGBRegressor
import pickle

st.set_page_config(page_title="Price calculator", page_icon=":bar_chart:", layout="wide")

@st.cache_data
def get_data_from_excel():
    dfsok = pd.read_csv(
        "Filled2.csv")
    return dfsok

@st.cache_data
def get_column_names():
    columns = joblib.load("columns.sav")
    return columns

@st.cache_resource
def get_model():
    #rf=pickle.load(open("pima.pickle.dat", "rb"))
    rf = joblib.load("rf3.pkl")
    #rf = joblib.load("rf1.pkl")
    return rf

columns=get_column_names()
dfsok = get_data_from_excel()
rf=get_model()

def predict(df2):
    df2['concat'] = df2['márka'] + df2['típus']
    df2.drop([ 'márka', 'típus'], axis=1, inplace=True)
    categorical_cols = ['üzemanyag', 'kivitel', 'concat', 'hajtás']
    df2 = pd.get_dummies(df2, columns = categorical_cols).reindex(columns=columns, fill_value=0)

    #X_cols = df2.loc[:, df2.columns != Y_col].columns
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.model_selection import train_test_split, cross_val_predict, KFold, cross_val_score
    from sklearn.ensemble import AdaBoostRegressor
    # X_train, X_test, y_train, y_test = train_test_split(df_scaled[X_cols], df_scaled[Y_col],test_size=0.2, random_state=42)
    #X_train, X_test, y_train, y_test = train_test_split(df2[X_cols], df2[Y_col], test_size=0.1, random_state=42)
    if not isinstance(rf, (XGBRegressor, RandomForestRegressor)):
        raise ValueError("Model is not of type RandomForestRegressor or ExtraTreesRegressor")

    prediction = rf.predict(df2)
    return prediction

selected=option_menu(
    menu_title=None,
    options=['Predictor','Dashboard'],
    default_index=0,
    orientation='horizontal'
)


car_makers = {
    'Toyota': ['Corolla', 'Camry', 'RAV4', 'Highlander', 'Sienna'],
    'Honda': ['Civic', 'Accord', 'CR-V', 'Pilot', 'Odyssey'],
    'Nissan': ['Altima', 'Sentra', 'Maxima', 'Rogue', 'Pathfinder'],
    'Ford': ['F-150', 'Mustang', 'Escape', 'Explorer', 'Expedition']
}

# Create an empty DataFrame with columns 'Car Maker' and 'Car Model'
df2 = pd.DataFrame(columns=['Car Maker', 'Car Model'])

# Loop through the car makers and their car models, and add them to the DataFrame
for maker, models in car_makers.items():
    for model in models:
        # Add the example to the DataFrame
        df2 = df2.append({'Car Maker': maker, 'Car Model': model}, ignore_index=True)

kinyert_adat = []
kinyert_extra=[]
markak=df2['Car Maker'].unique().tolist()

if selected=='Predictor':
    st.title("Car price predictor")
    markak=sorted(dfsok["márka"].unique().tolist())
    marka = st.sidebar.selectbox('Select car maker', (markak))
    tipusok = sorted(list(set(dfsok["típus"].loc[dfsok["márka"] == marka])))
    tipus = st.sidebar.selectbox('Select car type', (tipusok))
    slider1_range = st.sidebar.slider('Year', 1990, 2021, 2000)
    teljesitmeny = st.sidebar.slider('Power', 0, 500, 0)
    hengerurtartalom = st.sidebar.slider('Cylinder capacity', 0, 5000, 0)
    uzemanyag = st.sidebar.selectbox('Fuel type', options=['Benzin', 'Dízel', 'Elektromos', 'Gáz', 'Hibrid'])
    Kivitel = st.sidebar.selectbox('Design',
                                   options=['Cabrio', 'Coupe', 'Egyterű', 'Ferdehátú', 'Kisbusz', 'Kombi', 'Pickup',
                                            'Sedan', 'Terepjáró', 'Városi terepjáró'])
    allapot = st.sidebar.selectbox('Condition', options=['Kitűnő', 'Megkímélt', 'Normál', 'Sérülésmentes', 'Újszerű'])
    hajtas = st.sidebar.selectbox('Drive', options=['Front wheel drive', 'Rear wheel drive', 'All wheel drive'])
    valto_input = st.sidebar.selectbox('Select gearbox type', options=['manual', 'automatic'])
    if valto_input == 'manual':
        valto = 0
    else:
        valto = 1

    suly = st.sidebar.slider('Weight', 0, 5000, 0)
    use_range = st.sidebar.checkbox('Use a range of values for Slider 2')


    if use_range:
        slider2_range = st.sidebar.slider('Mileage Range', 0, 500000, (0, 10000),1000)
    else:
        slider2 = st.sidebar.slider('Mileage', 0, 500000, 0,1000)

    if marka != "" and tipus != '':
        disable_state = False
    else:
        disable_state = True

    keresett_extrak=["tolatóradar","USB","GPS",'tempomat','sávtartó rendszer','fűthető első ülés','tolatókamera','kulcsnélküli indítás','multifunkciós kormánykerék',
                     'távolságtartó tempomat','esőszenzor','Apple CarPlay','Android Auto','bluetooth-os kihangosító']
    options = st.sidebar.multiselect('What extras are included', keresett_extrak)


    submit_button = st.button(label="Let's predict",disabled=disable_state)

    if submit_button:
        # Add a list of values in increments of 10 between the range of slider1_range


        #st.write(kinyert_adat)
        if use_range:
            # Create a dataframe with the values of the list as one column and the simple slider value in every row

            slider1_list = list(range(slider2_range[0], slider2_range[1] + 1, 10000))

            # Create a dataframe with the values of the list as one column and the simple slider value in every row

            kinyert_adat.append(marka)
            kinyert_adat.append(tipus)
            kinyert_adat.append(slider1_range)
            kinyert_adat.append(valto)
            kinyert_adat.append(teljesitmeny)
            kinyert_adat.append(hengerurtartalom)
            kinyert_adat.append(uzemanyag)
            kinyert_adat.append(Kivitel)
            kinyert_adat.append(allapot)
            kinyert_adat.append(hajtas)
            kinyert_adat.append(suly)
            for extra in keresett_extrak:
                if any(extra in word for word in options):
                    kinyert_extra.append(1)
                else:
                    kinyert_extra.append(0)

            data = {'márka': kinyert_adat[0], 'típus': kinyert_adat[1], 'teljesítmény': kinyert_adat[4],
                    'évjárat': kinyert_adat[2],
                    'sebességváltó': kinyert_adat[3],
                    'üzemanyag': kinyert_adat[6], 'kivitel': kinyert_adat[7], 'állapot': kinyert_adat[8],
                    'km._óra_állás': slider1_list, 'hengerűrtartalom': kinyert_adat[5],
                    'hajtás': kinyert_adat[9], 'saját_tömeg': kinyert_adat[10]}

            df = pd.DataFrame(data)
            rownumber=len(df)
            #st.write(rownumber)
            df_extra=pd.DataFrame(data=[kinyert_extra]*rownumber,columns=keresett_extrak)
            #df_extra2=pd.concat([df_extra]*rownumber, ignore_index=True)
            df_osszevont = df.join(df_extra, how='left')
            #st.write(kinyert_adat)
            #st.write(df_osszevont)
            predictions=predict(df_osszevont)
            #st.write(predictions)
            #fig = plt.figure(figsize=(10, 4))
            #sns.lineplot(y=predictions,x=df_osszevont['km._óra_állás'])
            fig_car_prices = px.line(
                #sales_by_product_line,
                y=predictions,
                x=df_osszevont['km._óra_állás'],
                orientation="h",
                title="<b>Price range according to the selected parameters</b>",
                color_discrete_sequence=["#0083B8"] * len(df_osszevont['km._óra_állás']),
                template="plotly_white",
                width=100, height=350


            ).update_layout(xaxis_title="Mileage", yaxis_title="Price")


            #st.pyplot(fig)
            st.plotly_chart(fig_car_prices, use_container_width=True)



        else:

            kinyert_adat.append(marka)
            kinyert_adat.append(tipus)
            kinyert_adat.append(slider1_range)
            kinyert_adat.append(slider2)
            kinyert_adat.append(valto)
            kinyert_adat.append(teljesitmeny)
            kinyert_adat.append(hengerurtartalom)
            kinyert_adat.append(uzemanyag)
            kinyert_adat.append(Kivitel)
            kinyert_adat.append(allapot)
            kinyert_adat.append(hajtas)
            kinyert_adat.append(suly)
            # st.write(kinyert_adat)
            for extra in keresett_extrak:
                if any(extra in word for word in options):
                    kinyert_extra.append(1)
                else:
                    kinyert_extra.append(0)

            data = {'márka': kinyert_adat[0], 'típus': kinyert_adat[1], 'teljesítmény': kinyert_adat[5],
                    'évjárat': kinyert_adat[2], 'sebességváltó': kinyert_adat[4],
                    'üzemanyag': kinyert_adat[7], 'kivitel': kinyert_adat[8], 'állapot': kinyert_adat[9],
                    'km._óra_állás': kinyert_adat[3], 'hengerűrtartalom': kinyert_adat[6],
                    'hajtás': kinyert_adat[10], 'saját_tömeg': kinyert_adat[11]}
            df = pd.DataFrame(data, index=[0])
            #st.write(kinyert_extra)
            df_extra=pd.DataFrame(data=[kinyert_extra],columns=keresett_extrak)
            df_osszevont=df.join(df_extra,how='left')
            #st.write(df_osszevont)
            ár = int(round((predict(df_osszevont))[0],0))
            #st.write(ár)
            #predicted2=predict(df_osszevont)
            st.markdown(f"The predicted price of this type of vehicle is **{ár:,}** Forint")
            # st.write(kinyert_adat)
if selected == 'Dashboard':
    st.sidebar.header("Please Filter Here:")
    minden_gyarto=st.sidebar.checkbox('Every manufacturer',value=True)
    if minden_gyarto:
        gyarto=sorted(dfsok["márka"].unique())
    else:
        gyarto = st.sidebar.multiselect(
            "Select the Manufacturer:",
            options=sorted(dfsok["márka"].unique()),
            default=sorted(dfsok["márka"].explode().value_counts().index[:10].tolist())
        )
    translation_dict = {
        'Diesel': 'Dízel',
        'Petrol': 'Benzin',
        'Electric': 'Elektromos',
        'Gas': 'Gáz',
        'Front-wheel drive':'Első kerék',
        'Rear-wheel drive': 'Hátsó kerék',
        'All-wheel drive': 'Összkerék',
        # Add more translations as needed
    }


    fuel = st.sidebar.multiselect(
        "Select the Fuel:",
        options=['Diesel','Petrol','Electric','Gas'],
        default=['Diesel','Petrol','Electric','Gas']
    )
    type = st.sidebar.multiselect(
        "Select the car type:",
        options=sorted(dfsok["kivitel"].unique()),
        default=sorted(dfsok["kivitel"].unique()),
    )
    drive = st.sidebar.multiselect(
        "Select the car drive:",
        options=['Front-wheel drive','Rear-wheel drive','All-wheel drive'],
        default=['Front-wheel drive','Rear-wheel drive','All-wheel drive'],
    )
    selected_fuel = [translation_dict.get(option, option) for option in fuel]
    selected_drive = [translation_dict.get(option, option) for option in drive]

    try:
        df_selection = dfsok.query(
            " márka == @gyarto  & üzemanyag == @selected_fuel & kivitel ==@type & hajtás ==@drive"
        )


        st.markdown("##")

        average_price = int(round(df_selection["vételár"].mean(), 0))
        average_year = int(round(df_selection["évjárat"].mean(), 0))
        average_milage = int(round(df_selection["km._óra_állás"].mean(), 0))
        numberofcars=len(df_selection)

        left_column, middle_column, right_column,extra_column = st.columns(4)
        with left_column:
            st.subheader("Average price:")
            st.subheader(f"HUF  {average_price:,}")
        with middle_column:
            st.subheader("Average Production Year:")
            st.subheader(f"{average_year} ")
        with right_column:
            st.subheader("Average Milage (km):")
            st.subheader(f" {average_milage}")
        with extra_column:
            st.subheader("Number of cars:")
            st.subheader(f" {numberofcars}")

        st.markdown("""---""")

        sales_by_product_line = (
            df_selection.groupby('évjárat').size())



        fig_product_sales = px.line(
            sales_by_product_line,
            y=[0],
            x=sales_by_product_line.index,
            orientation="h",
            title="<b>Yearly distribution of advertised cars</b>",
            color_discrete_sequence=["#0083B8"] * len(sales_by_product_line),
            template="plotly_white",
            width=100, height=350,
            labels={
                     "évjárat": "Year"

                 }

        )
        fig_product_sales.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False)),
            showlegend=False

        )



        sales_by_hour = df_selection.groupby('márka').size().sort_values(ascending=False).head(10)

        fig_hourly_sales = px.bar(
            sales_by_hour,


            title="<b>Top 10 Manufacturer</b>",
            color_discrete_sequence=["#0083B8"] * len(sales_by_hour),
            template="plotly_white",
            width=100, height=350,
            labels={
                "márka": "Brand"

            }
        )
        fig_hourly_sales.update_layout(
            xaxis=dict(tickmode="linear"),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=(dict(showgrid=False)),
            showlegend=False

        )

        price_by_year = df_selection.groupby('évjárat')['vételár'].mean()
        fig_yearly_price = px.line(
            price_by_year,


            title="<b>Average price by year</b>",
            color_discrete_sequence=["#0083B8"] * len(sales_by_hour),
            template="plotly_white",
            width=100, height=350,
            labels={
                "évjárat": "Year"

            }
        )
        fig_yearly_price.update_layout(
            xaxis=dict(tickmode="linear"),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=(dict(showgrid=False)),
            showlegend=False

        )


        left_column, right_column = st.columns(2)
        left_column.plotly_chart(fig_hourly_sales, use_container_width=True)
        right_column.plotly_chart(fig_product_sales, use_container_width=True)
        left_column, right_column = st.columns(2)
        left_column.plotly_chart(fig_yearly_price, use_container_width=True)
        #right_column.plotly_chart(price_by_milage, use_container_width=True)


        #st.write(df_selection)
    except:
        st.write('You have to pick a value for every filter')