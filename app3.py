from fastapi import FastAPI, UploadFile, File, Header,Form
import os
import pandas as pd
from BL import bl as bl
from CLASSFN import segments
app = FastAPI()

# API for file upload
@app.post('/file_upload/')
async def upload_file(uid: str = Header(default=None, convert_underscores=False), file: UploadFile = File(...)):
    flag = True
    try:
        file_contents = await file.read()

        upload_folder =  r'c:/uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, f'{uid}'+"_"+file.filename)

        file_extension = file.filename.split(".")[-1]  # Get the original file extension
        full_filename = f'{uid}.{file_extension}'

        file_path = os.path.join(upload_folder, full_filename)
        with open(file_path, "wb") as f:
            f.write(file_contents)
        
        #csv_path = os.path.join(upload_folder, full_filename)
        #df = pd.read_csv(csv_path)      

        # Determine the file extension
        file_extension = file.filename.split(".")[-1].lower()

        if file_extension == 'csv':
            # Load a CSV file into a DataFrame
            df = pd.read_csv(file_path)
        elif file_extension in ['xls', 'xlsx']:
            # Load an Excel file into a DataFrame
            df = pd.read_excel(file_path)

    except Exception as e:
        flag = False 
        print("Error:", str(e))

    return {'file uploaded successfully': flag}


# API to run business logic on the uploaded file
@app.post('/run_business_logic/')
async def run_business_logic(uid: str = Header(default=None, convert_underscores=False)):
    try: 
        # if check is None:
        #     return {'error': 'The "check" parameter is required.'}

        df = bl.load_file_into_dataframe(uid)

        if df is not False:

            bl.table_creat(df)

            return {"Process":"Insertion Successful"}
        else:
            return {'error': f'{uid} is not found'}
    except Exception as e:
        return {'error': str(e)}



@app.post('/active_cust_page/')
async def active_customer(data: segments.ProcessedDataFrame, check :str):
    try:
        # Call run_business_logic to obtain processed_df
        processed_df = run_business_logic(data.processed_df)

        # Now you can use processed_df
        df_pivot = bl.customer_daily_rfm(processed_df, check)
        return df_pivot
    except Exception as e:
        return {'error': str(e)}



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
