
from pydantic import BaseModel
from typing import Optional,List
import json


####################################
class upload_file(BaseModel):
    uid :str
    file_path: str
    
class ProcessedDataFrame(BaseModel):
    processed_df: str

class Active_customer(BaseModel):
    id: str
    value: str
    file_path: str

class customer_segmentation(BaseModel):
    CUST_ID: int
    Recency:int
    Frequency:int
    Monetary:int
    Customer_segments:str

