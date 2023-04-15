#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
import pandas as pd
from datetime import timedelta
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

TOP_1M_DOMAINS = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'
TOP_1M_DOMAINS_FILE = 'top-1m.csv'

def get_data():
    top_doms = pd.read_csv(TOP_1M_DOMAINS)
    top_data = top_doms.to_csv(index=False)

    with open(TOP_1M_DOMAINS_FILE, 'w') as f:
        f.write(top_data)

def top_domain_count_zone():
    top_domain_count_zone = pd.read_csv(TOP_1M_DOMAINS_FILE, names=['rank', 'domain'])
    top_domain_count_zone['domain_zone'] = top_domain_count_zone['domain'].apply(lambda x: x.split('.')[-1])
    top_domain_count_zone = top_domain_count_zone.groupby('domain_zone',as_index = False).agg({'domain' : 'count'}).rename(columns={'domain' : 'count'}).sort_values('count', ascending = False).head(10)
    
    with open('top_domain_count_zone.csv', 'w') as f:
        f.write(top_domain_count_zone.to_csv(index=False, header=False))

def top_domain_max_len():
    top_domain_max_len = pd.read_csv(TOP_1M_DOMAINS_FILE, names=['rank', 'domain'])
    top_domain_max_len['len_domain'] = top_domain_max_len.domain.str.len()
    top_domain_max_len = top_domain_max_len.sort_values('len_domain', ascending = False).head(1)
    
    with open('top_domain_max_len.csv', 'w') as f:
        f.write(top_domain_max_len.to_csv(index=False, header=False))

def rank_airflow():
    rank_airflow = pd.read_csv(TOP_1M_DOMAINS_FILE, names=['rank', 'domain'])
    rank_airflow = df_2.query("domain == 'airflow.com'")[['rank']]
    
    with open('rank_airflow.csv', 'w') as f:
        f.write(rank_airflow.to_csv(index=False, header=False))
    
def print_answer(ds):
    with open('top_domain_count_zone.csv', 'r') as f:
        top_domain_count_zone = f.read()
    with open('top_domain_max_len.csv', 'r') as f:
        top_domain_max_len = f.read()
    with open('rank_airflow.csv', 'r') as f:
        rank_airflow = f.read()
    date = ds

    print(f'Top count domains for date {date}')
    print(top_domain_count_zone)

    print(f'Max len domain for date {date}')
    print(top_domain_max_len)
    
    print(f'rank airflow for date {date}')
    print(rank_airflow)
 
default_args = {
    'owner': 'v-mazur',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2023, 3, 3),
}
schedule_interval = '0 20 * * *'

dag = DAG('v-mazur_lesson_2', default_args=default_args, schedule_interval=schedule_interval)

t1 = PythonOperator(task_id='get_data',
                    python_callable=get_data,
                    dag=dag)

t2 = PythonOperator(task_id='top_domain_count_zone',
                                    python_callable=top_domain_count_zone,
                                    dag=dag)

t3 = PythonOperator(task_id='top_domain_max_len',
                                        python_callable=top_domain_max_len,
                                        dag=dag)

t4 = PythonOperator(task_id='rank_airflow',
                                        python_callable=rank_airflow,
                                        dag=dag)

t5 = PythonOperator(task_id='print_answer',
                    python_callable=print_answer,
                    dag=dag)

t1 >> [t2, t3, t4] >> t5


# In[ ]:




