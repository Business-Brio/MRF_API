# import pandas as pd
# import requests
# import pickle
# import datetime as dt
# #from mysql.connector import MySQLConnection
# import configparser
# import os

# def load_file_into_dataframe(uid, api_url):
#     try:
#         # Make a POST request to your API to get the DataFrame
#         headers = {'uid': uid}
#         response = requests.post(api_url, headers=headers)

#         if response.status_code == 200:
#             # The request was successful, parse the JSON response into a DataFrame
#             df = pd.read_json(response.content, orient='records')
#             return df
#         else:
#             # The request returned an error, return an error message
#             return {"Error": f"API returned status code {response.status_code}: {response.text}"}

#     except Exception as e:
#         return {"Error": str(e)}
#     #Example usage:
# api_url = 'http://127.0.0.1:8000/docs#/default/run_business_logic_run_business_logic__post/'  # Replace with the actual URL of your API
# uid = 'uid'  # Replace with the UID you want to fetch

# # Call the function to load the DataFrame from the API
# result = load_file_into_dataframe(uid, api_url)

# # Check if it's a DataFrame or an error message
# if isinstance(result, pd.DataFrame):
#     # You have a DataFrame, you can work with it here
#     print(result.head())
# else:
#     # There was an error, print the error message
#     print(result["Error"])

# def original_data(filepath):

#     df=pd.read_csv(filepath,low_memory=False,parse_dates=["InvoiceDate"])
#     df.rename(columns={"Customer_ID":"Customer ID"},inplace=True)
#     #df["InvoiceDate"]=pd.to_datetime(df["InvoiceDate"])
#     df.drop('Unnamed: 0',axis=1,inplace=True)
#     return df
# old azure
# db_params = {
#     'user': 'brioadmin@briomariadb.mariadb.database.azure.com',
#     'password': 'Gbsm@1234',
#     'host': 'briomariadb.mariadb.database.azure.com',  # Hostname or IP address of the MySQL server
#     'port': 3306,  # Port number of the MySQL server (usually 3306)
#     'database': 'cust_seg'


import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Database connection parameters
# db_params = {
#     'host': '127.0.0.1',
#     'port': 3306,
#     'database': 'test',
#     'user': 'root',
#     'password': 'Admin@1234'
# }
db_params = {
    'user': 'root',
    'password': 'admin',
    'host': '127.0.0.1',  # Hostname or IP address of the MySQL server
    'port': 3306,  # Port number of the MySQL server (usually 3306)
    'database': 'test'
}

def insert_data_into_mysql(table_name, data, if_exists='append', chunk_size=1000):
    try:
        # Create a database engine
        engine = create_engine(f"mysql+mysqlconnector://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")

        # Configure chunk size for data insertion
        chunk_size = chunk_size  # Adjust this based on your server's capabilities and data size

        # Insert data into the database using chunks
        for chunk_start in range(0, len(data), chunk_size):
            chunk = data.iloc[chunk_start:chunk_start + chunk_size]

            try:
                # Insert the chunk into the database
                chunk.to_sql(name=table_name, con=engine, if_exists=if_exists, index=False)
            except SQLAlchemyError as e:
                print(f"Error while inserting data: {e}")

        # Commit the transaction and close the connection
        engine.dispose()
    except Exception as e:
        print(f"An error occurred: {e}")



# Load your data into a DataFrame (assuming you have already loaded the data)
#data = pd.read_csv('E:/Cust Seg/Price_data.csv', low_memory=False)

# Call the function to insert data into the database table
#insert_data_into_mysql('actual_data', data, if_exists='append', chunk_size=1000)
