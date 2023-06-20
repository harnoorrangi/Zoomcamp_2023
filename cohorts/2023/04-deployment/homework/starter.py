
import pickle
import pandas as pd
import os
import sys


#Load saved model
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)



#Read and prepare data
def read_data(filename):
    
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    return df



#Create dictionary and prepare data for modelling. 
def prepare_data(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    return X_val


#Apply model and save the results
def apply_model(input_file, output_file,year,month):
    df = read_data(input_file)
    x_val = prepare_data(df)
    y_pred = model.predict(x_val)
    print(y_pred.mean())
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['predictions'] = y_pred
    df_result = df[['ride_id','predictions']]
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression='None',
        index=False
    )


def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = 'results.parquet'
    
    apply_model(input_file=input_file, output_file=output_file,year=year, month=month)  
    

if __name__ == '__main__':
    run()



