import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import base64
from io import BytesIO
from base64 import b64encode
from dash import html
import plotly.graph_objects as go
#from DAL import dal as dl
import os
import pandas as pd
from DAL import dal as dl

# Define the directory where your files are located
file_directory =  r'c:/uploads'

def find_file_by_uid(uid):
    """
    Find a file in the specified directory that contains the given UID in its name.

    Args:
        uid (str): The UID to search for in file names.

    Returns:
        str or None: The path to the found file or None if not found.
    """
    for filename in os.listdir(file_directory):
        if uid in filename:
            return os.path.join(file_directory, filename)
    return None

def compare_dataframe_columns(df):
    try:
        # Define the predefined column names
        predefined_columns = ["CUST_ID", "VISIT_ID", "SKU_ID", "SKU_DESC", "VOLUME", "VISIT_DT"]

        # Get the column names from the DataFrame and sort them
        df_columns = df.columns.tolist()
        df_columns.sort()

        # Sort the predefined columns for consistent comparison
        predefined_columns.sort()

        # Compare the sorted column names
        return df_columns == predefined_columns
    except ValueError as e:
        return False
            

def load_file_into_dataframe(uid):
    """
    Load a file with the given UID in its name into a DataFrame.

    Args:
        uid (str): The UID to search for in file names.

    Returns:
        pd.DataFrame or None: The loaded DataFrame or None if the file is not found.
    """
    file_path = find_file_by_uid(uid)
    #print(file_path)
    if file_path:
        try:
            # Load the file into a DataFrame (assuming it's a CSV file)
            df = pd.read_csv(file_path)

            # Check if the columns of df1 and df2 are the same
            result = compare_dataframe_columns(df)

            return df
        except Exception as e:
            return False

# # Example usage:
# uid_to_find = "your_uid_here"  # Replace with the UID you're looking for
# loaded_df = load_file_into_dataframe(uid_to_find)

# if loaded_df is not None:
#     # DataFrame loaded successfully, you can now work with it
#     print("DataFrame loaded successfully.")
#     print(loaded_df.head())
# else:
#     print("File not found or couldn't be loaded.")


# Common function for writing fig across different dataframe
# def format_axes_common(fig,df):
#      """
#      ##Input Parameters##
#      fig=Figure Object
#      df=Dataframe

#      ##Output##
#      returns a fig component 
#      """
#      for i, ax in enumerate(fig.axes):
#         ax.text(0.5, 0.5, f"{df.iloc[i]}", va="center", ha="center")
#         #ax.text(0.5, 0.5, "ax%d" % (i+2), va="center", ha="center")
#         ax.tick_params(labelbottom=False, labelleft=False)
#         ax.set_title(f'{df.iloc[i].name}',
#         fontsize = 14, fontweight ='bold')
        
# def fig_componet_store(df,check="Daily"):
 
#     """
#      ##Input Parameters##
#      fig=Figure Object
#      df=Dataframe

#      ##Output##
#      returns a fig component 
#     """

#     if len(df)>0:
#         fig = plt.figure(layout="constrained")
#         gs = GridSpec(2, 2, figure=fig)
#         LFLM = fig.add_subplot(gs[0, 0])
#         # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
#         LFHM = fig.add_subplot(gs[0,1])
#         HFLM = fig.add_subplot(gs[1,0])
#         HFHM = fig.add_subplot(gs[1,1])
#         #fig.suptitle("GridSpec")

#         format_axes_common(fig,df)


#         # # Save it to a temporary buffer.
#         buf = BytesIO()
#         fig.savefig(buf, format="png")

#         # # Embed the result in the html output.
#         data = base64.b64encode(buf.getbuffer()).decode("ascii")
#     else:
#         fig = plt.figure()
#         ax = fig.add_subplot()
#         fig.subplots_adjust(top=0.85)
#         ax.axis([0, 10, 0, 10])
        
#         ax.text(3, 5, f'No Data For {check} Analysis', style='italic',
#         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
#         ax.tick_params(labelbottom=False, labelleft=False)

#         # # Save it to a temporary buffer.
#         buf = BytesIO()
#         fig.savefig(buf, format="png")

#         # # Embed the result in the html output.
#         data = base64.b64encode(buf.getbuffer()).decode("ascii")

#     return data
    

def active_customer_data(df_original):
    # Remove invalid Customer IDs
    df_original = df_original[df_original["CUST_ID"] > 0]

    # Convert datatypes
    df_original[["VISIT_ID","SKU_ID","SKU_DESC","DEPT_DESC"]] = df_original[["VISIT_ID","SKU_ID","SKU_DESC","DEPT_DESC"]].astype("string")
    df_original["CUST_ID"] = df_original["CUST_ID"].astype(int)

    # Removing null oberservations
    df_original.dropna(inplace=True)

    # Removing canceled orders and invalid quantities
    df_original = df_original[~df_original["VISIT_ID"].str.contains("C", na=False)]
    df_original = df_original[df_original["VOLUME"] > 0]

    # Add Total Price column
    df_original["TotalPrice"] = df_original["SALES"]

    # Add separate Date column
    df_original['Date'] = pd.to_datetime(df_original['VISIT_DT']).dt.date
    # Defining today date as max(InvoiceDate) + 2 days
    today_date = dt.date (2023, 4, 2)
    print(f" Maximum invoice date: {df_original.VISIT_DT.max()} \n Today date: {today_date}")

    df = df_original.copy()
    df['year']=pd.PeriodIndex(df.Date,freq='Y')
    df['quarter'] = pd.PeriodIndex(df.Date, freq='Q')
    df['month']=pd.PeriodIndex(df.Date, freq='M')
    df['week']=pd.PeriodIndex(df.Date, freq='W')
    df['day']=pd.PeriodIndex(df.Date, freq='D')

    df_pivot = df.pivot_table(index='CUST_ID', columns='week', aggfunc={'Date':'nunique'},fill_value=0)
    df_pivot.columns = df_pivot.columns.droplevel(0) #remove date
    df_pivot.columns.name = None #remove week
    for i in df_pivot.index:
        for j in df_pivot.columns:
            if df_pivot.loc[i,j] >= 1:
             break
            else:
             df_pivot.loc[i][j] = -1

    # Replace -1 with NaN
    df_pivot = df_pivot.replace(-1,np.nan)
    perc_25 = np.nanquantile(df_pivot, 0.25, axis=1)
    df_perc_25 = pd.DataFrame()
    df_perc_25["CUST_ID"] = df_pivot.index
    df_perc_25["Perc_25"] = perc_25
    df_daily = df_perc_25[df_perc_25["Perc_25"] >1.0]

    # weekly customer
    df_nondaily = df_perc_25[df_perc_25["Perc_25"] <=1.0]
    df_nondaily.drop(columns = {"Perc_25"}, axis=1, inplace=True)
    df = df[df['CUST_ID'].isin(df_nondaily["CUST_ID"])]
    df_pivot = df.pivot_table(index='CUST_ID', columns='month', aggfunc={'Date':'nunique'},fill_value=0)
    df_pivot.columns = df_pivot.columns.droplevel(0) #remove date
    df_pivot.columns.name = None
    for i in df_pivot.index:
      for j in df_pivot.columns:
        if df_pivot.loc[i,j]>=1:
         break
        else:
         df_pivot.loc[i][j] =-1

    df_pivot=df_pivot.replace(-1,np.nan)
    perc_25=np.nanquantile(df_pivot, 0.25, axis=1)
    df_perc_25=pd.DataFrame()
    df_perc_25["CUST_ID"]=df_pivot.index
    df_perc_25["Perc_25"]=perc_25
    df_weekly = df_perc_25[df_perc_25["Perc_25"] >= 2.0]

    # Monthly customer
    df_nonweekly = df_perc_25[df_perc_25["Perc_25"] < 2.0]
    df_nonweekly.drop(columns = {"Perc_25"}, axis=1, inplace=True)
    df = df[df['CUST_ID'].isin(df_nonweekly["CUST_ID"])]
    df_pivot = df.pivot_table(index='CUST_ID', columns='quarter', aggfunc={'Date':'nunique'},fill_value=0)
    df_pivot.columns = df_pivot.columns.droplevel(0) #remove date
    df_pivot.columns.name = None
    for i in df_pivot.index:
      for j in df_pivot.columns:
        if df_pivot.loc[i,j]>=1:
         break
        else:
          df_pivot.loc[i][j] =-1

    df_pivot=df_pivot.replace(-1,np.nan)
    perc_25=np.nanquantile(df_pivot, 0.25, axis=1)
    df_perc_25=pd.DataFrame()
    df_perc_25["CUST_ID"]=df_pivot.index
    df_perc_25["Perc_25"]=perc_25
    df_monthly = df_perc_25[df_perc_25["Perc_25"] >=2.0]

    # Quaterly customer
    df_nonmonthly = df_perc_25[df_perc_25["Perc_25"] < 2.0]
    df_nonmonthly.drop(columns = {"Perc_25"}, axis=1, inplace=True)
    df = df[df['CUST_ID'].isin(df_nonmonthly["CUST_ID"])]
    df_pivot = df.pivot_table(index='CUST_ID', columns='year', aggfunc={'Date':'nunique'},fill_value=0)
    df_pivot.columns = df_pivot.columns.droplevel(0) #remove date
    df_pivot.columns.name = None
    for i in df_pivot.index:
     for j in df_pivot.columns:
      if df_pivot.loc[i,j]>=1:
       break
     else:
       df_pivot.loc[i][j] =-1

    df_pivot=df_pivot.replace(-1,np.nan)
    perc_25=np.nanquantile(df_pivot, 0.25, axis=1)
    df_perc_25=pd.DataFrame()
    df_perc_25["CUST_ID"]=df_pivot.index
    df_perc_25["Perc_25"]=perc_25
    df_quarterly = df_perc_25[df_perc_25["Perc_25"] >=3.0]

    # Calculating recency, monetary, frequency and tenure metrics
    rfm = df_original.groupby("CUST_ID").agg({"Date":lambda date: (today_date - date.max()).days,
                                        "VISIT_ID": lambda num: num.nunique(),
                                          "TotalPrice": lambda price: price.sum()}) #total price per customer

    #rfm.columns = rfm.columns.droplevel(0)
    rfm.columns = ['Recency', 'Frequency', "Monetary"]
    df_daily.drop(columns = {"Perc_25"}, axis=1, inplace=True)
    rfm_daily=rfm[rfm["Recency"]<30]
    df_daily_active= rfm_daily[rfm_daily.index.isin(df_daily["CUST_ID"])]

    rfm_daily_inactive=rfm[rfm["Recency"]>30]
    df_daily_inactive = rfm_daily_inactive[rfm_daily_inactive.index.isin(df_daily.CUST_ID)]
    
    q2_F=df_daily_active["Frequency"].quantile(0.50)
    q2_M = df_daily_active["Monetary"].quantile(0.50)
    # create a list of our conditions
    conditions = [
        (df_daily_active['Frequency'] <= q2_F) & (df_daily_active['Monetary']<=q2_M),
        (df_daily_active['Frequency'] <= q2_F) & (df_daily_active['Monetary']>q2_M),
        (df_daily_active['Frequency']>q2_F) & (df_daily_active['Monetary']<=q2_M),
        (df_daily_active['Frequency']>q2_F) & (df_daily_active['Monetary']>q2_M)
        ]

    # create a list of the values we want to assign for each condition
    values = ['LFLM', 'LFHM', 'HFLM', 'HFHM']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df_daily_active['Customer_segments'] = np.select(conditions, values)
    df_daily_active_fn=df_daily_active.groupby('Customer_segments').median().rename(columns= {'Recency':"Median Recency","Frequency":"Median Frequency","Monetary":'Median Monetary'}).join(df_daily_active.Customer_segments.value_counts().rename('count'))
    
    df_weekly.drop(columns = {"Perc_25"}, axis=1, inplace=True)
    rfm_weekly_active=rfm[rfm["Recency"]<60]
    df_weekly_active= rfm_weekly_active[rfm_weekly_active.index.isin(df_weekly["CUST_ID"])]

    rfm_weekly_inactive=rfm[rfm["Recency"]>=60]
    df_weekly_inactive = rfm_weekly_inactive[rfm_weekly_inactive.index.isin(df_weekly.CUST_ID)]
    q2_F=df_weekly_active["Frequency"].quantile(0.50)
    q2_M = df_weekly_active["Monetary"].quantile(0.50)
    # create a list of our conditions
    conditions = [
        (df_weekly_active['Frequency'] <= q2_F) & (df_weekly_active['Monetary']<=q2_M),
        (df_weekly_active['Frequency'] <= q2_F) & (df_weekly_active['Monetary']>q2_M),
        (df_weekly_active['Frequency']>q2_F) & (df_weekly_active['Monetary']<=q2_M),
        (df_weekly_active['Frequency']>q2_F) & (df_weekly_active['Monetary']>q2_M)
        ]

    # create a list of the values we want to assign for each condition
    values = ['LFLM', 'LFHM', 'HFLM', 'HFHM']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df_weekly_active['Customer_segments'] = np.select(conditions, values)
    df_weekly_active_fn=df_weekly_active.groupby('Customer_segments').median().rename(columns= {'Recency':"Median Recency","Frequency":"Median Frequency","Monetary":'Median Monetary'}).join(df_weekly_active.Customer_segments.value_counts().rename('count'))

    df_monthly.drop(columns = {"Perc_25"}, axis=1, inplace=True)
    rfm_monthly_active=rfm[rfm["Recency"]<90]
    df_monthly_active= rfm_monthly_active[rfm_monthly_active.index.isin(df_monthly["CUST_ID"])]

    rfm_monthly_inactive=rfm[rfm["Recency"]>=90]
    df_monthly_inactive= rfm_monthly_inactive[rfm_monthly_inactive.index.isin(df_monthly.CUST_ID)]
    q2_F=df_monthly_active["Frequency"].quantile(0.50)
    q2_M = df_monthly_active["Monetary"].quantile(0.50)
    # create a list of our conditions
    conditions = [
        (df_monthly_active['Frequency'] <= q2_F) & (df_monthly_active['Monetary']<=q2_M),
        (df_monthly_active['Frequency'] <= q2_F) & (df_monthly_active['Monetary']>q2_M),
        (df_monthly_active['Frequency']>q2_F) & (df_monthly_active['Monetary']<=q2_M),
        (df_monthly_active['Frequency']>q2_F) & (df_monthly_active['Monetary']>q2_M)
        ]

    # create a list of the values we want to assign for each condition
    values = ['LFLM', 'LFHM', 'HFLM', 'HFHM']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df_monthly_active['Customer_segments'] = np.select(conditions, values)
    df_monthly_active_fn=df_monthly_active.groupby('Customer_segments').median().rename(columns={"Recency":"Median Recency","Frequency":"Median Frequency","Monetary":"Median Monetary"}).join(df_monthly_active.Customer_segments.value_counts().rename('count'))
    
    df_quarterly.drop(columns = {"Perc_25"}, axis=1, inplace=True)
    rfm_quarterly_active=rfm[rfm["Recency"]<180]
    df_quarterly_active = rfm_quarterly_active[rfm_quarterly_active.index.isin(df_quarterly["CUST_ID"])]

    rfm_quarterly_inactive=rfm[rfm["Recency"]>=180]
    df_quarterly_inactive = rfm_quarterly_inactive[rfm_quarterly_inactive.index.isin(df_quarterly["CUST_ID"])]

    q2_F=df_quarterly_active["Frequency"].quantile(0.50)
    q2_M = df_quarterly_active["Monetary"].quantile(0.50)
    # create a list of our conditions
    conditions = [
        (df_quarterly_active['Frequency'] <= q2_F) & (df_quarterly_active['Monetary']<=q2_M),
        (df_quarterly_active['Frequency'] <= q2_F) & (df_quarterly_active['Monetary']>q2_M),
        (df_quarterly_active['Frequency']>q2_F) & (df_quarterly_active['Monetary']<=q2_M),
        (df_quarterly_active['Frequency']>q2_F) & (df_quarterly_active['Monetary']>q2_M)
        ]

    # create a list of the values we want to assign for each condition
    values = ['LFLM', 'LFHM', 'HFLM', 'HFHM']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df_quarterly_active['Customer_segments'] = np.select(conditions, values)

    df_quarterly_active_fn=df_quarterly_active.groupby('Customer_segments').median().rename(columns={"Recency":"Median Recency","Frequency":"Median Frequency","Monetary":"Median Monetary"}).join(df_quarterly_active.Customer_segments.value_counts().rename('count'))

    return df_daily_active,df_daily_inactive,df_daily_active_fn,df_weekly_active,df_weekly_inactive,df_weekly_active_fn,df_monthly_active,df_monthly_inactive,df_monthly_active_fn,df_quarterly_active,df_quarterly_inactive,df_quarterly_active_fn


def elasticity(df_original):
    df_original = df_original[df_original["VOLUME"] > 0]
    df_original["Unit_price"] = df_original["SALES"]/df_original["VOLUME"]
    df_Stock = df_original.groupby(['VISIT_DT','SKU_ID']).agg({'Unit_price':'mean','VOLUME': 'mean' }).reset_index()
    x_pivot = df_Stock.pivot(index= 'VISIT_DT' ,columns='SKU_ID' ,values='Unit_price')
    x_values = pd.DataFrame(x_pivot.to_records())
    y_pivot = df_Stock.pivot( index = 'VISIT_DT',columns='SKU_ID', values='VOLUME')
    y_values = pd.DataFrame(y_pivot.to_records())
    points = []
    results_values = {
        "name": [],
        "price_elasticity": [],
        "price_mean": [],
        "quantity_mean": [],
        "intercept": [],
        "t_score":[],
        "slope": [],
        "coefficient_pvalue" : [],
    }
    #Append x_values with y_values per same product name
    for column in x_values.columns[1:]:
        column_points = []
        for i in range(len(x_values[column])):
            if not np.isnan(x_values[column][i]) and not np.isnan(y_values[column][i]):
                column_points.append((x_values[column][i], y_values[column][i]))
        df = pd.DataFrame(list(column_points), columns= ['x_value', 'y_value'])


        #Linear Regression Model
        import statsmodels.api as sm
        x_value =df['x_value']
        y_value = df['y_value']
        X = sm.add_constant(x_value)
        model = sm.OLS(y_value, X)
        result = model.fit()
        #(Null Hypothesis test) Coefficient with a p value less than 0.05
        if result.f_pvalue < 0.05:

            rsquared = result.rsquared
            coefficient_pvalue = result.f_pvalue
            intercept,slope = result.params
            mean_price = np.mean(x_value)
            mean_quantity = np.mean(y_value)
            tintercept,t_score = result.tvalues

            #Price elasticity Formula
            price_elasticity = (slope)*(mean_price/mean_quantity)
            #price_elasticity = (slope)*(mean_quantity/mean_price)

            #Append results into dictionary for dataframe
            results_values["name"].append(column)
            results_values["price_elasticity"].append(price_elasticity)
            results_values["price_mean"].append(mean_price)
            results_values["quantity_mean"].append(mean_quantity)
            results_values["intercept"].append(intercept)
            results_values['t_score'].append(t_score)
            results_values["slope"].append(slope)
            results_values["coefficient_pvalue"].append(coefficient_pvalue)

    final_df = pd.DataFrame.from_dict(results_values)
    df_elasticity = final_df[['name','price_elasticity','t_score','coefficient_pvalue','slope','price_mean','quantity_mean','intercept']]
    return df_elasticity

def product_rank(df_elasticity, values_column):

    #Divergent plot
    df=df_elasticity.copy()
    df['ranking'] = df[values_column].rank( ascending = True).astype(int)
    df.sort_values(values_column, ascending =False, inplace = True)

    #Adjust Ranking column and print dataframe
    pd.set_option('display.width', 4000)
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]

    df = df.iloc[:,:3]
    df.set_index('ranking', inplace=True)
    print(df)

def price_elasticity(df_original):
   
   df_elasticity=elasticity(df_original)
   pe_rank = product_rank(df_elasticity, 'price_elasticity')
   return pe_rank

def table_creat (df_original):
   
#    df_daily_active,df_daily_inactive,df_daily_active_fn,df_weekly_active,df_weekly_inactive,df_weekly_active_fn,df_monthly_active,df_monthly_inactive,df_monthly_active_fn,df_quarterly_active,df_quarterly_inactive,df_quarterly_active_fn =active_customer_data(df_original)
#    dl.insert_data_into_mysql('daily_active_cust', df_daily_active, if_exists='replace', chunk_size=3000)
#    dl.insert_data_into_mysql('daily_inactive_cust', df_daily_inactive, if_exists='replace', chunk_size=3000)
#    dl.insert_data_into_mysql('daily_active_group', df_daily_active_fn, if_exists='replace', chunk_size=3000)
#    dl.insert_data_into_mysql('weekly_active_cust', df_weekly_active, if_exists='replace', chunk_size=3000)
#    dl.insert_data_into_mysql('weekly_inactive_cust', df_weekly_inactive, if_exists='replace', chunk_size=3000)
#    dl.insert_data_into_mysql('weekly_active_group', df_weekly_active_fn, if_exists='replace', chunk_size=3000)
#    dl.insert_data_into_mysql('monthly_active_cust', df_monthly_active, if_exists='replace', chunk_size=3000)
#    dl.insert_data_into_mysql('monthly_inactive_cust', df_monthly_inactive, if_exists='replace', chunk_size=3000)
#    dl.insert_data_into_mysql('monthly_active_group', df_monthly_active_fn, if_exists='replace', chunk_size=3000)
#    dl.insert_data_into_mysql('quarterly_active_cust', df_quarterly_active, if_exists='replace', chunk_size=3000)
#    dl.insert_data_into_mysql('quarterly_inactive_cust', df_quarterly_inactive, if_exists='replace', chunk_size=3000)
#    dl.insert_data_into_mysql('quarterly_active_group', df_quarterly_active_fn, if_exists='replace', chunk_size=3000)

   pe_rank = product_rank(df_original)
   dl.insert_data_into_mysql('price_elasticity_list', pe_rank, if_exists='replace', chunk_size=3000)

   