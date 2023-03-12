#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from tqdm.auto import tqdm


# In[2]:


# Подгрузим данные и посмотрим на них
df_hw = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Module_statistic/hw_aa.csv', sep = ';')


# In[3]:


df_hw.info()


# In[4]:


# Проведем А/А тест

n = 100000
simulations = 1000
n_s = 1000
res = []

# Запуск симуляций A/A теста
for i in tqdm(range(simulations)):
    s1 = df_hw.query('experimentVariant == 1')['purchase'].sample(n_s, replace = False).values
    s2 = df_hw.query('experimentVariant == 0')['purchase'].sample(n_s, replace = False).values
    res.append(stats.ttest_ind(s1, s2, equal_var = False)[1]) # сохраняем pvalue

plt.hist(res, bins = 50)
plt.style.use('ggplot')
plt.xlabel('pvalues')
plt.ylabel('frequency')
plt.title("Histogram of ttest A/A simulations ")
plt.show()

# Проверяем, что количество ложноположительных случаев не превышает альфа
sum(np.array(res) <0.05) / simulations


# In[5]:


# Вывод: FPR > 0.05, что говорит о некорректности распедленеия данных в АА тесте. Найдем причину:


# In[6]:


df_hw.groupby('version', as_index = False).agg({'purchase' : 'mean'}) # Видно, что в версии 2.8.0 самый низкий показатель purchase. Посморим, может его и больше в какой-либо группе.


# In[7]:


df_hw.groupby(['version', 'experimentVariant'], as_index=False).agg({'purchase' : 'mean'}) # Действительно, больше всего его в 0 группе в сранении с 1


# In[8]:


# Исключим версию 2.8.0 и проведем повторный А/А тест


# In[9]:


# Проведем А/А тест с исключенеим версии 2.8.0 

n = 100000
simulations = 1000
n_s = 1000
res = []

# Запуск симуляций A/A теста
for i in tqdm(range(simulations)):
    s1 = df_hw.query("experimentVariant == 1 & version != 'v2.8.0'")['purchase'].sample(n_s, replace = False).values
    s2 = df_hw.query("experimentVariant == 0 & version != 'v2.8.0'")['purchase'].sample(n_s, replace = False).values
    res.append(stats.ttest_ind(s1, s2, equal_var = False)[1]) # сохраняем pvalue

plt.hist(res, bins = 50)
plt.style.use('ggplot')
plt.xlabel('pvalues')
plt.ylabel('frequency')
plt.title("Histogram of ttest A/A simulations ")
plt.show()

# Проверяем, что количество ложноположительных случаев не превышает альфа
sum(np.array(res) <0.05) / simulations


# In[10]:


# Отлично! FPR < 0.05, что говорит о том, что А/А тест сработал корректно (а также о том, что с версией 2.8.0 что-то не так, возможная техническая ошибка).


# In[ ]:




