from datasets import load_dataset
import pandas as pd
import os 

def load_truthful_qa(DATA_BASE):
    ds_path = os.path.join(DATA_BASE,"truthful_qa.jsonl")
    if(os.path.exists(ds_path)):
        df = pd.read_json(ds_path,lines=True)
    else:
        ds = load_dataset("truthful_qa",'generation')["validation"]
        df = ds.to_pandas()
        df.to_json(ds_path,lines=True,orient='records')
    return df

def load_med_qa(DATA_BASE):
    ds_path = os.path.join(DATA_BASE,"med_qa.jsonl")
    if(os.path.exists(ds_path)):
        df = pd.read_json(ds_path,lines=True)
    else:
        ds = load_dataset('openlifescienceai/medqa')["test"]
        df = ds.to_pandas()
        details_df = pd.DataFrame(ds['data'].tolist())
        ds = pd.concat([ds.drop('data', axis=1), details_df], axis=1)
        ## Make Corrected DF with only necessary columns
        corrected_df = {}
        corrected_df['id'] = ds['id'].values
        corrected_df['correct_answers'] = ds['Correct Answer'].values 
        corrected_df['question'] = ds['Question']
        df = pd.DataFrame(corrected_df)
        df.to_json(ds_path,lines=True,orient='records')
    return df

def load_fin_qa(DATA_BASE):
    ds_path = os.path.join(DATA_BASE,"med_qa.jsonl")
    if(os.path.exists(ds_path)):
        df = pd.read_json(ds_path,lines=True)
    else:
        ds = load_dataset('TheFinAI/FINQA_test', split='test')
        df = ds.to_pandas()
        df = df.rename(columns={'Open-ended Verifiable Question': 'question', 'Ground-True Answer': 'correct_answers'})
        df.to_json(ds_path,lines=True,orient='records')
    return df