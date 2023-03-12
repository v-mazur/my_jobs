#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Подгрузим данные и обработаем
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
from scipy.stats import norm
import scipy as st
plt.style.use('ggplot')

df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Module_statistic/hw_bootstrap.csv', sep = ';')


# In[2]:


df = df.rename(columns = {'Unnamed: 0' : 'Number'})


# In[3]:


df.info()


# In[4]:


df['value'] = df['value'].str.replace(',','.')


# In[5]:


df = df.astype({"value": float})


# In[6]:


# Посмотрим на кол-во показателей: их по 500, кол-во выборок больше 30 (это хорошо!)
df.query("experimentVariant == 'Control'").shape


# In[7]:


df.query("experimentVariant == 'Treatment'").shape


# In[8]:


# Посмотрим на нормальность распределения:


# In[9]:


st.stats.normaltest(df.query("experimentVariant=='Treatment'").value) # Распределение ненормально (p < 0.05)


# In[10]:


st.stats.normaltest(df.query("experimentVariant=='Control'").value) # Распределение нормально (p > 0.05)


# In[11]:


# Учитывая, что normaltest в Treatment показал ненормальность, посторим по нему гистограмму (заодно и по Control):


# In[12]:


df_Treatment = df.query("experimentVariant=='Treatment'")
sns.distplot(df_Treatment.value) # Да, он действительно ненормален...(присутсвуют выбросы)


# In[13]:


df_control = df.query("experimentVariant=='Control'")
sns.distplot(df_control.value) # Распределение нормальное


# In[14]:


# Данные проанализировали, теперь проведем A/B-тест


# In[15]:


# Проведем через t-test: Вывод: pvalue < 0.05, принимаем альтернативную гипотезу, т.е.изменения есть
st.stats.ttest_ind(df_Treatment.value                     ,df_control.value)


# In[16]:


# Проверяем через U-тест: Вывод: pvalue > 0.05, оставляем нулевую гипотезу, т.е.изменений нет
st.stats.mannwhitneyu(df_Treatment.value, df_control.value)#, alternative='two-sided')


# In[17]:


# Проверяем через бутстрап:
df_value_Treatment = df_Treatment.value
df_value_Control = df_control.value


# In[18]:


def get_bootstrap_2(
    df_value_Treatment, # числовые значения первой выборки
    df_value_Control, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1_1 = df_value_Treatment.sample(
            len(df_value_Treatment), 
            replace = True # параметр возвращения
        ).values
        
        samples_2_2 = df_value_Control.sample(
            len(df_value_Control), 
            replace = True
        ).values
        
        boot_data.append(statistic(samples_1_1)-statistic(samples_2_2)) # mean() - применяем статистику
        
    pd_boot_data = pd.DataFrame(boot_data)
        
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    for bar in bars:
        if bar.get_x() <= quants.iloc[0][0] or bar.get_x() >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else: 
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')
    
    plt.style.use('ggplot')
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}


# In[19]:


booted_data = get_bootstrap_2(df_value_Treatment, df_value_Control, statistic = np.median) # применяем медиану, тк присутсвуют выбросы


# In[20]:


booted_data["p_value"] # альфа


# In[21]:


booted_data["quants"] # ДИ


# In[22]:


# Вывод: на основе данной выборке эксперименты не отличаются, т.е. различий нет (т.к. 0 внутри ДИ, а pvalue > 0.05)


# In[23]:


# Общий вывод: 1. На полученный вывод через t-test я бы не стал ссылаться, учитывая тот факт, что распрделение у одной из групп ненормальное.
                  # Я бы стал опираться на выводы через U-тест и бутстрап, т.к. применяя данные методы, мы можем принебречь ненормальностью одной из групп.
# Таким образом, считаю, что правильный ответ: на основе данной выборки, мы не можем отвегнуть нулевую гиптоезу!

# При этом, стоит отметить, что можно провести дополнительный анализ через t-test, если отбросить выбросы. Прведем данный анализ.


# In[24]:


df_Treatment_norm = df.query("experimentVariant=='Treatment'")


# In[25]:


df_Treatment_norm.describe() # Видно, что большинство данных не больше 100. Откинем оставльное


# In[26]:


df_Treatment_norm.query("value > 100") # Видно, что большинство данных не больше 100. Откинем оставльное


# In[27]:


df_Treatment_norm = df_Treatment_norm.query("value < 100")


# In[28]:


df_Treatment_norm.info()


# In[29]:


st.stats.normaltest(df_Treatment_norm.value) # Распределение нормальное. Построим гистограмму на вскякий случай, чтобы визуально это увидеть


# In[30]:


sns.distplot(df_Treatment_norm.value)


# In[31]:


# Теперь проведем t-test. Вывод: на базе данной выборки принимаем нулевую гипотезу: изменений нет, т.к. pvalue > 0.05
st.stats.ttest_ind(df_Treatment_norm.value                     ,df_control.value)


# In[ ]:




