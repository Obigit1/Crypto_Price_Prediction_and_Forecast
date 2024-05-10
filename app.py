import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pyarrow as pa
from matplotlib import rcParams
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.impute import SimpleImputer
import warnings
import squarify

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_option('deprecation.showPyplotGlobalUse', False)
# Hide warnings
warnings.filterwarnings("ignore")

# Load data 
@st.cache_data
def load_data(selected_crypto=None):
        cryptos = ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "ADA-USD", "DOT1-USD", "XLM-USD", "LINK-USD",
                "BNB-USD", "DOGE-USD", "SOL1-USD", "USDC-USD", "UNI3-USD", "WBTC-USD", "ATOM1-USD", "AVAX-USD",
                "LUNA1-USD", "FIL-USD", "TRX-USD", "XEM-USD", "MIOTA-USD", "XTZ-USD", "AAVE-USD", "EOS-USD",
                "FTT-USD", "ALGO-USD", "XMR-USD", "NEO-USD", "MKR-USD"]
        data = yf.download(cryptos, start="2023-01-01", end="2024-01-01")
        new_data = data.dropna(axis=1, how='all')
        close_data = new_data['Close']
        if selected_crypto:
            close_data = close_data[[selected_crypto]]
        close_data['MIOTA-USD'] = close_data['MIOTA-USD'].interpolate()
        return close_data



def fetch_crypto_news():
    # This is a placeholder function to demonstrate how you can fetch cryptocurrency news.
    # You can replace this with your actual implementation to fetch news from an API or another source.
    return [
        {"headline": "Bitcoin Trades Around $57K", "source": "CoinDesk", "link": "https://www.coindesk.com/markets/2024/05/01/bitcoin-sinks-below-58k-crypto-market-drops-9-in-run-up-to-fed-decision/?_gl=1*17u99i5*_up*MQ..*_ga*MTMxNzM2MjgxOC4xNzE0NTg5MDIz*_ga_VM3STRYVN8*MTcxNDU4OTAyMy4xLjAuMTcxNDU4OTAyMy4wLjAuODA4MjA0NDMx"},
        {"headline": "Ethereum sets Record Profit", "source": "CryptoSlate", "link": "https://cryptoslate.com/tether-reports-record-4-52-billion-profit-in-q1-despite-shrinking-market-share/"},
        {"headline": "Binance withdraws application", "source": "The Block", "link": "https://www.theblock.co/post/266656/binance-withdraws-application-to-manage-abu-dhabi-investment-fund-report"},
        # Add more news articles from different sources
    ]

def display_crypto_news_feed():
    #st.markdown("<style> .news-container { position: fi; top: 40px; right: 40px; width: 200px; max-height: 300px; overflow-y: auto; border: 1px solid #ccc; border-radius: 5px; padding: 10px; } </style>", unsafe_allow_html=True)
    st.markdown("###### Cryptocurrency News Feed", unsafe_allow_html=True)
    
    # Fetch crypto news
    crypto_news = fetch_crypto_news()
    
    # Display the news feed container
    st.markdown("<div class='news-container'>", unsafe_allow_html=True)
    st.markdown("<ul style='list-style-type: none; padding: 0;'>", unsafe_allow_html=True)
    
    # Display fetched crypto news
    for news_item in crypto_news:
        st.markdown(
            f"<li><a href='{news_item['link']}' target='_blank'>{news_item['headline']}</a> - {news_item['source']}</li>",
            unsafe_allow_html=True
        )
    
    st.markdown("</ul>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Call the function to display the crypto news feed
display_crypto_news_feed()





def load_data1(selected_crypto1=None):
    cryptos = ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "ADA-USD", "DOT1-USD", "XLM-USD", "LINK-USD",
               "BNB-USD", "DOGE-USD", "SOL1-USD", "USDC-USD", "UNI3-USD", "WBTC-USD", "ATOM1-USD", "AVAX-USD",
               "LUNA1-USD", "FIL-USD", "TRX-USD", "XEM-USD", "MIOTA-USD", "XTZ-USD", "AAVE-USD", "EOS-USD",
               "FTT-USD", "ALGO-USD", "XMR-USD", "NEO-USD", "MKR-USD"]
    data1 = yf.download(cryptos, start="2023-01-01", end="2024-01-01")
    new_data1 = data1.dropna(axis=1, how='all')
    close_data1 = new_data1['Close']
    
    if selected_crypto1 and selected_crypto1 in close_data1.columns:
        close_data1[selected_crypto1] = close_data1[selected_crypto1].interpolate()

    return close_data1



    # Download historical data for the past year
def load_historical_data(selected_cryptos=None):
        chosen_cryptos = ["BTC-USD", "ETH-USD", "MKR-USD", "BNB-USD"]
        start_date = "2019-02-01"
        end_date = "2024-02-01"
        selected_crypto_dataset = yf.download(chosen_cryptos, start=start_date, end=end_date)

        close_historical_data = selected_crypto_dataset['Close']
        if selected_cryptos:
            close_historical_data = close_historical_data[[selected_cryptos]]
            return close_historical_data


def load_historical_data2(selected_cryptos2=None):
    chosen_cryptos2 = ["BTC-USD", "ETH-USD", "MKR-USD", "BNB-USD"]
    start_date = "2019-02-01"
    end_date = "2024-02-01"
    selected_crypto_dataset2 = yf.download(chosen_cryptos2, start=start_date, end=end_date)

    print("Selected Crypto Dataset:", selected_crypto_dataset2)  # Debugging statement

    if selected_crypto_dataset2 is None:
        print("Error: selected_crypto_dataset is None")  # Debugging statement
        return None

    close_historical_data2 = selected_crypto_dataset2['Close']
    if selected_cryptos2:
        close_historical_data2 = close_historical_data2[selected_cryptos2]
    return close_historical_data2




def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Welcome", "PCA / Cluster", "Selected Cryptos", "Top 4 Correlation",
                                           "EDA","Prophet" ,"LSTM Model" , "SVR", "ARIMA", "Investment Calculator"])

    if page == "Welcome":
        st.markdown("## Welcome to Cryptocurrency Predictive & Forecast System Demo")
        st.markdown("### APPLIED AI IN BUSINESS (COM724)  Assessment")
        st.markdown("#### OBINNA G. UGWUEGBU (Q102104484)")
        st.write("Use the navigation bar on the left to explore Selected Cryptos, PCA/Cluster analysis, Correlation analysis, Exploratory Data Analysis and the Models.")
        #display_crypto_news_feed()

    elif page == "PCA / Cluster":
        st.title("Principal Component Analysis (PCA) and K-Means Clustering")
        st.write("Performing PCA and K-Means clustering...")
        #display_crypto_news_feed()

        # Load data
        close_data = load_data()

        # Transpose the data for dimensionality reduction
        transposed_data = close_data.T

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=10)
        reduced_data = pca.fit_transform(transposed_data)

        # Create a DataFrame with reduced data
        reduced_df = pd.DataFrame(data=reduced_data, index=transposed_data.index,
                                  columns=[f'PC{i + 1}' for i in range(reduced_data.shape[1])])

        # Perform K-Means clustering
        num_clusters = 4
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans_model.fit(reduced_df)
        cluster_labels = kmeans_model.labels_
        reduced_df['Cluster'] = cluster_labels

        # Display the DataFrame with PCA and cluster labels
        st.write("PCA and Clustered Data:")
        st.write(reduced_df)

    
    elif page == "Selected Cryptos":
        st.title("Selected Cryptocurrencies")
        st.write("Displaying selected cryptocurrencies...")
        #display_crypto_news_feed()

        # Load data
        close_data = load_data()

        # Select BNB-USD, BTC-USD, ETH-USD, and MKR-USD from each cluster
        selected_cryptos = ['BNB-USD', 'BTC-USD', 'ETH-USD', 'MKR-USD']

        # Create a DataFrame containing only the selected cryptocurrencies
        selected_data = close_data[selected_cryptos]

        # Display the DataFrame
        st.write(selected_data)
    
    elif page == "Top 4 Correlation":
        st.title("Top 4 Positive & Negative Pair Correlation")
        st.write("Calculating correlation...")
        #display_crypto_news_feed()

        # List the selected cryptocurrencies
        selected_cryptos = ['BNB-USD', 'BTC-USD', 'ETH-USD', 'MKR-USD']
        # Display selectbox for chosen tickers
        selected_crypto = st.sidebar.selectbox("Select cryptocurrency", selected_cryptos)
        # Load historical data
        close_data = load_data()
        #close_data2 = load_historical_data2()

        

        # Function to get top correlated pairs
        def top_correlated_pairs(corr_matrix, crypto):
            # Get correlations for the specified cryptocurrency
            correlations = corr_matrix[crypto]

            # Exclude the cryptocurrency itself
            correlations = correlations[correlations.index != crypto]

            # Get the top 4 positive and negative correlations
            top_positive = correlations.nlargest(4)
            top_negative = correlations.nsmallest(4)

            return top_positive, top_negative

        # Calculate correlation matrix
        corr_matrix = close_data.corr()

        # Allow user to select a cryptocurrency
        #selected_crypto = input("Enter the cryptocurrency ticker to display top correlated pairs: ").upper()

        # Check if selected cryptocurrency is in the list of available tickers
        if selected_crypto in corr_matrix.index:
            # Display top correlated pairs for the selected cryptocurrency
            print(f"\nTop correlated pairs for {selected_crypto}:\n")

            # Positive correlations
            top_positive, top_negative = top_correlated_pairs(corr_matrix, selected_crypto)

            #print("Top Positive Correlations:")
            #print(top_positive)

            #print("\nTop Negative Correlations:")
            #print(top_negative)
        #else:
            #print("Selected cryptocurrency not found in the dataset.")
        #if all(crypto in close_data2.columns for crypto in selected_cryptos2):
            # Calculate the correlation matrix for the selected cryptocurrencies
           # selected_corr_matrix = close_data2[selected_cryptos2].corr()

            # Function to get top correlated pairs
            #def top_correlated_pairs(corr_matrix, crypto):
                # Get correlations for the specified cryptocurrency
             #   correlations = corr_matrix[crypto]

                # Exclude the cryptocurrency itself
               # correlations = correlations[correlations.index != crypto]

                # Get the top 4 positive and negative correlations
                #top_positive = correlations.nlargest(4)
                #top_negative = correlations.nsmallest(4)

               # return top_positive, top_negative

            # Display top correlated pairs for each cryptocurrency
            #for crypto in selected_cryptos2:
              #  st.write(f"\nTop correlated pairs for {crypto}:\n")

                # Positive correlations
               # top_positive, top_negative = top_correlated_pairs(selected_corr_matrix, crypto)

            st.write("Top Positive Correlations:")
            st.write(top_positive)

            st.write("\nTop Negative Correlations:")
            st.write(top_negative)    
        else:
            st.write("One or more selected cryptocurrencies not found in the dataset.")

  

    #elif page == "Correlation Heatmap":
        #st.title("Correlation Analysis")
        #st.write("Calculating correlations...")

        # Load data
        #close_data = load_data()

        # Select chosen cryptocurrencies
        #chosen_cryptos = ['BNB-USD', 'BTC-USD', 'ETH-USD', 'MKR-USD']

        # Extract the Close columns from each ticker
        #selected_crypto_dataset = close_data[chosen_cryptos]

        # Calculate correlation matrix
        #correlation_matrix = selected_crypto_dataset.corr()

        # Display the correlation matrix
        #st.write("Correlation Matrix:")
        #st.write(correlation_matrix)

        # Plot correlation heatmap
        #plt.figure(figsize=(8, 6))
        #sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.3)
        #plt.title("Cryptocurrency Close Price Correlation Heatmap")
        #st.pyplot()


    elif page == "EDA":
        st.title(" Exploratory Data Analysis of Selected Cryptos")
        #display_crypto_news_feed()

        # Define chosen tickers
        chosen_tickers = ["BTC-USD", "ETH-USD", "MKR-USD", "BNB-USD"]

        # Display selectbox for chosen tickers
        selected_crypto1 = st.sidebar.selectbox("Select cryptocurrency", chosen_tickers)

        # Load data
        close_data1 = load_data1(selected_crypto1)

        # Plot time series for selected cryptocurrency
        plt.figure(figsize=(12, 8))
        plt.plot(close_data1.index, close_data1[selected_crypto1], label=selected_crypto1)
        plt.title(f'{selected_crypto1} Close Price')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        st.pyplot()

        # Plot histogram for selected cryptocurrency
        plt.figure(figsize=(12, 8))
        plt.hist(close_data1[selected_crypto1], bins=50, alpha=0.7)
        plt.title(f'Histogram of {selected_crypto1} Prices')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.grid(True)
        st.pyplot()

        # Plot boxplot for selected cryptocurrency
        plt.figure(figsize=(12, 8))
        close_data1[selected_crypto1].plot(kind='box')
        plt.title(f'Boxplot of {selected_crypto1} Prices')
        plt.ylabel('Price')
        plt.grid(True)
        st.pyplot()

        # Plot area chart for selected cryptocurrency
        plt.figure(figsize=(12, 8))
        plt.fill_between(close_data1.index, close_data1[selected_crypto1], label=selected_crypto1, alpha=0.5)
        plt.title(f'Area Chart of {selected_crypto1} Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        st.pyplot()

         # Load data
        close_data = load_data()

        # Define chosen_data
        chosen_data = close_data[chosen_tickers]

        # Aggregate cryptocurrency prices for a specific date (e.g., the last date in the dataset)
        total_prices = chosen_data.iloc[-1].sum()

        # Calculate the proportion of each cryptocurrency's price within the total
        proportions = chosen_data.iloc[-1] / total_prices

        # Plot pie chart
        plt.figure(figsize=(8, 8))

        # Ensure proportions is a 1D array
        if isinstance(proportions, np.ndarray) and proportions.ndim == 1:
            plt.pie(proportions, labels=[selected_crypto], autopct='%1.1f%%', startangle=140)
        else:
            # If proportions is not a 1D array, use the column names as labels
            plt.pie(proportions, labels=proportions.index, autopct='%1.1f%%', startangle=140)

        plt.title('Distribution of Cryptocurrency Prices')
        st.pyplot()


        # Calculate the total price for each ticker
        total_prices = chosen_data.sum()

        # Plot treemap
        plt.figure(figsize=(10, 5))
        squarify.plot(sizes=total_prices, label=total_prices.index, alpha=0.7)
        plt.axis('off')
        plt.title('Cryptocurrency Treemap')
        st.pyplot()


        # Plot pairplot
        #pairplot = sns.pairplot(chosen_data)
        #pairplot.fig.suptitle("Pairplot of Selected Cryptocurrencies", y=1.02)  # Set title above the figure
        #st.pyplot()
        
    elif page == "LSTM Model":
        st.title("LSTM Model for Cryptocurrency Price Prediction")
        st.write("Performing LSTM Model...")
        #display_crypto_news_feed()

        selected_cryptos = st.sidebar.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "MKR-USD", "BNB-USD"])

        # Load data for the selected cryptocurrency
        close_historical_data = load_historical_data(selected_cryptos)
        
        # LSTM Model code...
        scaled_df = close_historical_data

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(scaled_df)

        # Define the number of time steps
        time_steps = 10

        # Create sequences
        def create_sequences(data, time_steps):
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[i:(i + time_steps)])
                y.append(data[i + time_steps])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data, time_steps)

        # Split data into train and test sets
        train_size = int(len(X) * 0.7)
        test_size = len(X) - train_size
        X_train, X_test = X[0:train_size], X[train_size:len(X)]
        y_train, y_test = y[0:train_size], y[train_size:len(X)]

        # Reshape input data for LSTM model
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], scaled_df.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], scaled_df.shape[1]))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(units=scaled_df.shape[1]))
        model.compile(optimizer='adam', loss='mse')

        # Train LSTM model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1,
                            shuffle=False)

        # Plot loss curves
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        st.pyplot()

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Inverse scaling
        train_pred_inv = scaler.inverse_transform(train_pred)
        y_train_inv = scaler.inverse_transform(y_train)
        test_pred_inv = scaler.inverse_transform(test_pred)
        y_test_inv = scaler.inverse_transform(y_test)

        

        # Plot historical and predicted prices for the selected cryptocurrency
        plt.plot(np.arange(len(y_train_inv)), y_train_inv.flatten(), 'g', label="Historical")
        plt.plot(np.arange(len(y_train_inv), len(y_train_inv) + len(y_test_inv)), y_test_inv.flatten(), label="True")
        plt.plot(np.arange(len(y_train_inv), len(y_train_inv) + len(y_test_inv)), test_pred_inv.flatten(), 'r',
                 label="Predicted")
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        st.pyplot()

        plt.plot(y_test_inv.flatten(), marker='.', label="true")
        plt.plot(test_pred_inv.flatten(), 'r', label="prediction")
        plt.ylabel('BTC-USD')
        plt.xlabel('Time Step')
        plt.legend()
        st.pyplot();
    

    elif page == "Prophet":
        #elif page == "Prophet":
        # Code for the "Prophet" page
        st.title("Prophet Model for Cryptocurrency Price Prediction")
        st.write("Performing Prophet Model...")
        #display_crypto_news_feed()

        selected_cryptos = st.sidebar.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "MKR-USD", "BNB-USD"], key="cryptocurrency_selectbox")

        # Load data for the selected cryptocurrency
        close_historical_data = load_historical_data(selected_cryptos)

        # Function to train Prophet model
        def train_prophet_model(close_historical_data, forecast_days):
            model = Prophet()
            model.fit(close_historical_data)
            future_dates = model.make_future_dataframe(periods=forecast_days)
            prediction = model.predict(future_dates)
            return model, prediction

        # Add input for forecast days
        forecast_days = st.sidebar.number_input("Enter the number of forecast days:", value=60, min_value=1)

        # Prepare data for Prophet model
        selected_col = close_historical_data[[selected_cryptos]].rename(columns={selected_cryptos: 'y'})
        selected_col['ds'] = selected_col.index

        # Train Prophet model
        model, prediction = train_prophet_model(selected_col, forecast_days)

        # Define signal function
        def get_signal(predictions):
            if predictions.iloc[-1]['yhat'] > predictions.iloc[-2]['yhat']:
                return "BUY"
            elif predictions.iloc[-1]['yhat'] < predictions.iloc[-2]['yhat']:
                return "SELL"
            else:
                return "HOLD"

        # Calculate signal
        current_signal = get_signal(prediction)

        # Display signal in sidebar
        st.sidebar.subheader("Current Signal")
        st.sidebar.write(current_signal)

        # Plot predictions using Plotly
        st.subheader("Prophet Forecast Plot")
        fig = model.plot(prediction)
        st.pyplot(fig)

        st.subheader("Prophet Components Plot")
        fig2 = model.plot_components(prediction)
        st.pyplot(fig2)



        
    #elif page == "SVR":
        #st.title("Support Vector Regression (SVR) Model for Cryptocurrency Price Prediction")
        #st.write("Performing SVR Model...")

        # Select cryptocurrency
        #selected_cryptos = st.sidebar.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "MKR-USD", "BNB-USD"])

        
        # Load data for the selected cryptocurrency
        #close_historical_data = load_historical_data(selected_cryptos)

                
        #def train_svr_model(close_historical_data, prediction_days, chosen_crypto):
            # Create a column shifted n units up
         #   close_historical_data['Prediction'] = close_historical_data[chosen_crypto].shift(-prediction_days)
         #   X = np.array(close_historical_data.drop(['Prediction'], 1))
          #  X = X[:len(close_historical_data) - prediction_days]
           # y = np.array(close_historical_data['Prediction'])
            #y = y[:-prediction_days]
            #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
            #svr_rbf.fit(x_train, y_train)
            #return svr_rbf, x_test, y_test


         # Define prediction days
        #prediction_days = st.sidebar.number_input("Enter the number of forecast days:", value=30, min_value=1)

        # Train SVR model
        #model, x_test, y_test = train_svr_model(close_historical_data, prediction_days, selected_cryptos)

        # Make predictions
        #svm_prediction = model.predict(x_test)

        # Convert prediction dates to datetime objects
        #prediction_dates = pd.date_range(start=close_historical_data.index[-prediction_days], periods=len(svm_prediction))

        # Define actual values (y)
       # y_actual = y_test[-len(svm_prediction):]

        # Plot actual vs. predicted values
        #plt.figure(figsize=(10, 6))
        #plt.plot(prediction_dates, y_actual, label='Actual', color='blue', marker='o')
        #plt.plot(prediction_dates, svm_prediction, label='Predicted', color='red', marker='x')

       # plt.title('Actual vs. Predicted Prices for ' + selected_cryptos)
        #plt.xlabel('Date')
        #plt.ylabel('Price (USD)')
        #plt.legend()

        #plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

        #plt.tight_layout()
        #st.pyplot(plt)

    elif page == "SVR":
        st.title("Support Vector Regression (SVR) Model for Cryptocurrency Price Prediction")
        st.write("Performing SVR Model...")
        #display_crypto_news_feed()

        # Select cryptocurrency
        selected_cryptos = st.sidebar.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "MKR-USD", "BNB-USD"])

        # Define prediction_days
        prediction_days = 20
        prediction_days = st.sidebar.number_input("Enter the number of forecast days:", value=30, min_value=1)
        # Load data for the selected cryptocurrency
        close_historical_data = load_historical_data(selected_cryptos)

        if close_historical_data is not None:  # Check if data was successfully loaded
            # Create a DataFrame with close historical data and prediction column
            btc_df = close_historical_data.copy()
            btc_df['Prediction'] = btc_df[selected_cryptos].shift(-prediction_days)

            # Prepare features (X) and target (y)
            X = np.array(btc_df.drop(['Prediction'], axis=1))
            X = X[:-prediction_days]
            y = np.array(btc_df['Prediction'])
            y = y[:-prediction_days]

            # Split the data into training and testing sets
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Prepare prediction days array
            prediction_days_array = np.array(btc_df.drop(['Prediction'], axis=1))[-prediction_days:]

            # Create and train the support Vector Machine (Regression) using radial basis function
            svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
            svr_rbf.fit(x_train, y_train)

            # Calculate confidence score
            svr_rbf_confidence = svr_rbf.score(x_test, y_test)

            # Perform predictions
            svm_prediction = svr_rbf.predict(x_test)
            svm_prediction_future = svr_rbf.predict(prediction_days_array)

            # Convert prediction dates to datetime objects
            prediction_dates = pd.to_datetime(btc_df.index[-prediction_days:])

            # Define signal function
            #def get_signal(predictions):
            #    if predictions[-1] > predictions[-2]:
            #        return "SELL"
            #    elif predictions[-1] < predictions[-2]:
            #        return "BUY"
            #    else:
            #        return "HOLD"

            # Display signal in sidebar
            #st.sidebar.subheader("Current Signal")
            #current_signal = get_signal(svm_prediction_future)
            #st.sidebar.write(current_signal)


            # Plot actual vs. predicted values
            plt.figure(figsize=(10, 6))
            plt.plot(prediction_dates, y[-prediction_days:], label='Actual', color='blue', marker='.')
            plt.plot(prediction_dates, svm_prediction_future, label='Predicted', color='red', marker='x')

            plt.title('Actual vs. Predicted BTC Prices')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()

            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

            plt.tight_layout()
            st.pyplot()
        else:
            st.write("Error: The selected cryptocurrency is not available.")


            


        
        # Streamlit app code
    elif page == "ARIMA":
        st.title("ARIMA Model for Cryptocurrency Price Prediction")
        st.write("Performing ARIMA Model...")
        #display_crypto_news_feed()
        
        selected_cryptos = st.sidebar.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "MKR-USD", "BNB-USD"])

        # Load data for the selected cryptocurrency
        close_historical_data = load_historical_data(selected_cryptos)

        # Define normalization function
        def normalize_data(data):
            scaler = StandardScaler()
            return scaler.fit_transform(data.values.reshape(-1, 1))

        # Split data into training and testing sets
        split_data = int(len(close_historical_data) * 0.7)
        training_data = close_historical_data[0:split_data][selected_cryptos]
        testing_data = close_historical_data[split_data:][selected_cryptos]

        # Normalize data
        normalized_training_data = normalize_data(training_data)
        normalized_testing_data = normalize_data(testing_data)

        # Train ARIMA model and make predictions
        model_predictions = []
        n_test_obser = len(normalized_testing_data)
        
        for i in range(n_test_obser):
            model = ARIMA(normalized_training_data, order=(4, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            model_predictions.append(yhat)
            actual_test_value = normalized_testing_data[i]
            normalized_training_data = np.append(normalized_training_data, [actual_test_value])

        # Inverse transform the predicted and actual values to the original scale
        scaler = StandardScaler()
        scaler.fit(close_historical_data[selected_cryptos].values.reshape(-1, 1))
        model_predictions = scaler.inverse_transform(np.array(model_predictions).reshape(-1, 1)).flatten()
        actual_values = scaler.inverse_transform(normalized_testing_data).flatten()

        # Plotting the actual and predicted values
        plt.figure(figsize=(10, 6))
        plt.grid()
        date_range = testing_data.index
        plt.plot(date_range, model_predictions, color='green', marker='.', linestyle='dashed', label='Predicted')
        plt.plot(date_range, actual_values, color='b', label='Actual')  # Plotting actual values
        plt.title(f'{selected_cryptos} Actual vs. Predicted Prices (ARIMA)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot()

    elif page == "Crypto News":
        # Display cryptocurrency news feed
        display_crypto_news_feed()  


    elif page == "Investment Calculator":
            st.title("Investment Calculator")
            st.write("Calculate potential earnings based on your investment")

            selected_cryptos = st.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "MKR-USD", "BNB-USD"])

            # Add input for forecast days
            forecast_days = st.number_input("Enter the number of forecast days:", value=60, min_value=1)

            # Add input for investment amount
            investment_amount = st.number_input("Enter your investment amount in USD:", value=1000.0, min_value=0.0)

            if st.button("Calculate"):
                # Load data for the selected cryptocurrency
                close_historical_data = load_historical_data(selected_cryptos)

                # Function to train Prophet model
                def train_prophet_model(close_historical_data, forecast_days):
                    model = Prophet()
                    model.fit(close_historical_data)
                    future_dates = model.make_future_dataframe(periods=forecast_days)
                    prediction = model.predict(future_dates)
                    return model, prediction

                # Prepare data for Prophet model
                selected_col = close_historical_data[[selected_cryptos]].rename(columns={selected_cryptos: 'y'})
                selected_col['ds'] = selected_col.index

                # Train Prophet model
                model, prediction = train_prophet_model(selected_col, forecast_days)

                # Calculate potential return
                predicted_price_today = prediction.iloc[-1]['yhat']
                predicted_price_tomorrow = prediction.iloc[-2]['yhat']

                investment_return = (predicted_price_tomorrow - predicted_price_today) * (investment_amount / predicted_price_today)

                # Calculate the number of units of cryptocurrency
                units_of_crypto = investment_amount / predicted_price_today

                st.write(f"Potential return: {investment_return:.2f} USD")
                st.write(f"Number of units of cryptocurrency: {units_of_crypto:.6f}")


        
# Run the app
if __name__ == "__main__":
    main()