import json
import pandas as pd
import dill
from datetime import datetime
import glob
import os

path = os.environ.get('PROJECT_PATH', '.')


#files = glob.glob('{path}data/models/cars_pipe*.pkl')
#files.sort(key=os.path.getmtime)
#file_for_model=files[0]

def predict():
    path = os.environ.get('PROJECT_PATH', '.')

    dirname = f'{path}/data/models/'
    files = os.listdir(dirname)
    with open(f'{path}/data/models/{files[0]}', 'rb') as file:
        model=dill.load(file)
    s=[]
    for filename in glob.glob(f'{path}data/test/*.json'):
            with open(filename, 'r', encoding='utf-8') as fin:
                form = json.load(fin)
                df = pd.DataFrame.from_dict([form])
                y = model.predict(df)
                x = {'car_id': df['id'][0], 'pred': y[0]}
                s.append(x)
    df = pd.DataFrame(s)
    df.to_csv(f'{path}data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv', sep = ',')








if __name__ == '__main__':
    predict()