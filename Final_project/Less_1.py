#!/usr/bin/env python
# coding: utf-8

# # Задание 1
# 
# ## Одной из основных задач аналитика в нашей команде является корректное проведение экспериментов. Для этого мы применяем метод A/B–тестирования. В ходе тестирования одной гипотезы целевой группе была предложена новая механика оплаты услуг на сайте, у контрольной группы оставалась базовая механика. В качестве задания Вам необходимо проанализировать итоги эксперимента и сделать вывод, стоит ли запускать новую механику оплаты на всех пользователей.
# 
# В качестве входных данных Вы имеете 4 csv-файла:
# 
# groups.csv - файл с информацией о принадлежности пользователя к контрольной или экспериментальной группе (А – контроль, B – целевая группа).
# 
# groups_add.csv - дополнительный файл с пользователями, который вам прислали спустя 2 дня после передачи данных.
# 
# active_studs.csv - файл с информацией о пользователях, которые зашли на платформу в дни проведения эксперимента. 
# 
# checks.csv - файл с информацией об оплатах пользователей в дни проведения эксперимента. 

# ### Импортируем нужные библиотеки, загрузим данные и проведем предварительное исследование

# In[1]:


import pandas as pd
import numpy as np
import scipy as st
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy import stats
from scipy.stats import chi2_contingency
from tqdm.auto import tqdm


groups = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Final_project/groups.csv', sep = ';')
group_add = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Final_project/group_add.csv')
active_studs = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Final_project/active_studs.csv')
checks = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Final_project/checks.csv', sep = ';')


# In[2]:


groups.head()


# In[3]:


group_add.head()


# In[4]:


active_studs.head()


# In[5]:


checks.head()


# ### Проверим, существуют ли пустые показатели. Вывод: пустых показателей нет, значит эксперимент прошел удачно

# In[6]:


groups.isna().sum()


# In[7]:


group_add.isna().sum()


# In[8]:


active_studs.isna().sum()


# In[9]:


checks.isna().sum()


# ### Проверим, все ли id индивидуальны. Вывод: да.

# In[10]:


groups_id = groups.id.nunique()
groups_shape = groups.shape[0]
print(groups_id==groups_shape)


# In[11]:


group_add_id = group_add.id.nunique()
group_add_shape = group_add.shape[0]
print (group_add_id == group_add_shape)


# In[12]:


active_studs_id = active_studs.student_id.nunique()
active_studs_shape = active_studs.shape[0]
print (active_studs_id == active_studs_shape)


# In[13]:


checks_id = checks.student_id.nunique()
checks_id_shape = checks.shape[0] 
print (checks_id == checks_id_shape)


# ### Рассмтрим данные из checks. Видим, что нам представлены данные только тех, кто непосредственно оплатил: rev > 0

# In[14]:


checks.rev.describe()


# ### Посмотрим, есть ли совпадения между group_add и groups. А то не понятно, что за второй файл, а спросить не у кого. Вывод: id пользователей разные (совпадений не найдено), а значит для дальнейшего решения просто скажем, что две недели назад нам просто забыли его отдать, либо произошел тех сбой.

# In[15]:


# Запишем все id в groups в список, а после отфильтруем в group_add
groups_id = groups.id.to_list()
group_add.query('id == @groups_id')


# ### Учитывая предыдущий вывод, нам необходимо соединить две таблицы в одну

# In[16]:


group_unity = pd.concat([groups, group_add]) #Используем concat, чтобы одну таблицу сделать под другой


# In[17]:


group_unity # проверим, все ли прошло удачно: в groups строк 74 484, в group_add 92, в group_unity 74 576 = (74 484 + 92). P.s. доверяй, но проверяй


# ### С первыми двумя таблицами ясно, ошибок нет (все id уникальны). Теперь я хочу проверить, все ли верно с таблицами active_studs и checks. Если человек заходил на сайт и не сделал покупку, то это нормальная ситуация. А вот если не заходил, но сделал покупку, то тут уже говорит нам о том, что была совершена ошибка при сборе данных. Проверим на ошибку сбора данных.

# In[18]:


active_studs_id_list = active_studs.student_id.to_list()
checks_pass_id = checks.query('student_id != @active_studs_id_list') 
checks_pass_id # вот наша и ошибочка при сборе данных (149 пользоваталей совершили покупку, при этом они не заходили на сайт)


# ### Примем решение: так как мы выявили ошибку при сборе данных: пользователь как будто не заходил, но покупка была совершена, то будем считать, что ошибка произошла в данных active_studs. Руководствовался следующей логикой: деньги нам поступили, они отражены в бух отчете, в отличии от active_studs, а значит пользователь точно оплачивал. Значит, для дальнейшего корректного анализа необходимо в active_studs добавить недостающих пользователей из checks.

# In[19]:


active_studs_full = pd.concat([active_studs, checks_pass_id]).drop(columns=['rev']) # Проверим кол-во строк:в active_studs 8341 строк, в checks_pass_id 149, в итоговом active_studs_full 8490 (=149+8341)
active_studs_full


# ### Теперь у нас есть файл, в котором указаны id тех, кто заходил на сайт. Их нужно вмерджить (соединить) в group_unity, чтобы понять, кто непосредственно из нашей выборки заходил на сайт (они же не были в курсе, что они в эксперименте и могли впринципе не заходить на наш сайт). Значит, для корректного анализа необходимо убрать те id, которые не зашли наш сайт в дни проведения теста.

# In[20]:


# Для начала проверим, что все id из active_studs есть в group_unity
active_studs_full_id = active_studs_full.student_id.to_list()
group_unity.query('id == @active_studs_full_id') # Вывод: кол-во строк с условием query 8341 совпавдает с кол-вом колонок active_studs_full. Это говорит о том, что теперь эксперимент можно считать удачным


# In[21]:


# В таком случае, вмердживаем, оставляя для дальнейшего анализа только тех, кто заходил на сайт
active_studs_full = active_studs_full.rename(columns = {'student_id' : 'id'}) # предварительно меняем название колонки для мерджа
data = active_studs_full.merge(group_unity, on = 'id', how = 'left')


# ### Теперь вмерджим в наш датасет c нужными нам id стоимоть покупок из checks

# In[22]:


checks = checks.rename(columns = {'student_id' : 'id'})
data_full = data.merge(checks, on = 'id', how = 'left').fillna(0)


# In[23]:


data_full # Наш датасет готов!


# ### Отлично, наши данные готовы. Теперь высчитаем метрики для наших данных, а именно: 
# #### - Конверсия в покупку (CR);
# #### - Средний чек (по сути это у нас и будет ARPPU)

# ## Конверсия в покупку (CR)

# In[24]:


# вычислим, сколько в прицнипе у нас id распределены в группах
cr = data_full.groupby('grp', as_index = False).agg({'id' : 'count'}).rename(columns = {'id' : 'count_id'})


# In[25]:


# вычислим, сколько у нас id, совершивших покупки
cr['pay_id'] = data_full.query('rev > 0').groupby('grp', as_index = False).agg({'id' : 'count'}).id


# In[26]:


cr['cr'] = round((cr.pay_id / cr.count_id * 100), 2) # Высчитываем CR


# In[27]:


cr_data = cr.drop(columns = ['count_id', 'pay_id']) # удалим уже ненужные колонки 


# In[28]:


cr_data # конверсия (CR) готова!


# ### Вывод: различия есть, но небольшие: разница составялет 6,83 - 6,27 = 0,56 (причем результат ухудшился). Далее, проверим, статистически значимы ли эти данные

# ### AB - тест для конверсии (CR).
# #### Так как мы смотрим на категориальные переменные, будем использовать ХИ - квадрат

# In[29]:


# Для начала разделим группу A и B и вычислим, сколько в одной и другой группе конвертировало в покупку
grp_A_cr_true = data_full.query("grp == 'A' & rev > 0").id.count() # - конвертировали в группе A
grp_B_cr_true = data_full.query("grp == 'B' & rev > 0").id.count() # - конвертировали в группе B

grp_A_cr_false = data_full.query("grp == 'A' & rev == 0").id.count() # - не конвертировали в группе A
grp_B_cr_false = data_full.query("grp == 'B' & rev == 0").id.count() # - не конвертировали в группе В


# In[30]:


grp_A_cr_true


# In[31]:


grp_B_cr_true


# In[32]:


grp_A_cr_false


# In[33]:


grp_B_cr_false


# In[34]:


# Создадим сводную таблицу

s1 = pd.Series({
    "A": grp_A_cr_true,
    "B": grp_B_cr_true
})
s2 = pd.Series({
    "A": grp_A_cr_false,
    "B": grp_B_cr_false
})

cr_test = pd.DataFrame({
    "commit": s1,
    "not_commit": s2
})


cr_test


# In[35]:


#Далее, проведем сам xи-тест
stat, p, dof, expected = chi2_contingency(cr_test)
  
alpha = 0.05
print("p-value is " + str(p))
if p <= alpha:
    print('Вывод: Есть основания отклонить нулевую гипотезу. Мы принимаем альтренативную гипотезу, т.к. тест показал, что связи существуют, различия есть')
else:
    print('Вывод: Нет оснований отклонить нулевую гипотезу. Мы принимаем нулевую гипотезу, т.к. тест показал, что связей не существуют, различий нет')


# 

# ## Средний чек

# In[36]:


check_mean = data_full.query('rev > 0').groupby('grp', as_index=False)             .agg({'rev':'mean'}).rename(columns={'rev':'mean_check'})
check_mean['mean_check'] = check_mean.mean_check.round(2)
check_mean # наблюдается улучшение показателя: в выборке B средний чек выше, чем в А.


# ### Вывод: наблюдается улучшение: средний чек в группе В, чем в А (на 248 руб). Проверим, статистически значимы ли эти изменения.

# ### AB - тест для среднего чека

# In[37]:


# Для начала разделим группу A и B 
gr_a_chek = data_full.query("grp == 'A' & rev > 0").rev
gr_b_chek = data_full.query("grp == 'B' & rev > 0 ").rev


# Заметим, что большая разнциа в абсолютном показателе между двумя сравниваемыми группам

# In[38]:


gr_b_chek.shape[0] - gr_a_chek.shape[0]


# ###### Так как мы рассматриваем непрерывные показатели, проведем тест на нормальность каждой группы выборки

# #### Для выборки А

# In[39]:


st.stats.normaltest(gr_a_chek)


# In[40]:


sns.distplot(gr_a_chek)


# ####  Вывод: распределение ненормально (p-value < 0.05)

# #### Для выборки B

# In[41]:


st.stats.normaltest(gr_b_chek)


# In[42]:


sns.distplot(gr_b_chek)


# #### Вывод: распределение ненормально (p-value < 0.05)

# ### Так как распределения ненормальны, а количество пользователей в двух группах значительно разные, то можно воспользоваться Буртстрапом для определения стат значимости:

# In[43]:


def get_bootstrap_2(
    gr_a_chek, # числовые значения первой выборки
    gr_b_chek, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1_1 = gr_a_chek.sample(
            len(gr_a_chek), 
            replace = True # параметр возвращения
        ).values
        
        samples_2_2 = gr_b_chek.sample(
            len(gr_b_chek), 
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


# In[44]:


booted_data = get_bootstrap_2(gr_a_chek, gr_b_chek) # видно, что 0 находится НЕ внутри ДИ (доверительного интервала), а значит мы принимаем альтернативную гипотезу: изменения есть


# In[45]:


booted_data["p_value"] # на всякий случай выведем p-value (p-value < 0.05, а значит, что принимаем альтернативную гипотезу: изменения есть)


# In[46]:


# Я бы также проверил бы данную гипотезу еще методом U-теста на всякий случай:
st.stats.mannwhitneyu(gr_a_chek, gr_b_chek) # Вывод: принимаем альтернативную гипотезу, т.к. p-value < 0,05: изменения есть


# ### Вывод: на основании вышеуказанных методов, мы можем сказать о том, что статистически значимые различия есть.

# 
# # ОБЩИЙ ВЫВОД: на основании проведенных А/В - тестов мы принимаем следующую гипотезу: статистически значимые различия между двумя группами экперимента сущетсвуют в части определения среднего чего. А значит, что группа В принесла нам больше денег, чем группа А, т.е. мы улучшили свои показатели. Можно распростарянять фичу.
# 
# # Предложения:
# ### - увеличить время эксперимента для более точного результата;
# ### - разработать другие фичи, помогающему нашему бизнесу и снова провести АВ - тест
# 
# # Доложить:
# ### - о не вхождении в датасет active_studs пользоватлей, которые совершили покупки согласно датасету checks (149 id)

# # -------------------------------------------------------------------------------------------

# # Задание 2
# 
# ## Задание 2.1.  Образовательные курсы состоят из различных уроков, каждый из которых состоит из нескольких маленьких заданий. Каждое такое маленькое задание называется "горошиной". Назовём очень усердным учеником того пользователя, который хотя бы раз за текущий месяц правильно решил 20 горошин.
# 

# In[47]:


# Подключаемся к Clickhouse

import pandahouse as ph

connection_default = {'host': 'http://clickhouse.beslan.pro:8080',
                      'database':'default',
                      'user':'student', 
                      'password':'dpo_python_2020'
                     }


# ### В задании не сказано, что выполнить 20 горошин должен только по одному предмету, поэтому в условии я этого не прописывал и ответ получил - 136. 
# 
# ### Однако, в  q_1_var_2 сделал на всякий случай и с разделением на предметы (т.е. выполнено 20 горошин только по конкретному предмету). Вдруг заказчик это имел ввиду. 

# In[48]:


# Без разделений на предметы:

q_1_var_1 = ''' 

SELECT COUNT (st_id) AS count_id
FROM 
(
SELECT st_id, toStartOfMonth(first_time) AS first_time, 
toStartOfMonth(last_time) AS last_time,
       dateDiff('month', first_time, last_time) AS diff_time
FROM (
SELECT 
    st_id,
    SUM ( correct ) AS sum_correct,
    MIN ( timest ) AS first_time,
    MAX ( timest ) AS last_time
FROM default.peas
GROUP BY ( st_id )
HAVING sum_correct >= 20 
)
WHERE diff_time < 1) '''

task_2_1_var_1  = ph.read_clickhouse(query=q_1_var_1, connection=connection_default)
task_2_1_var_1.count_id.item() # Выводим ответ.

# Вообще, можно было было заметить, что все студенты сделали все в рамках одного месяца, 
# и в запрос можно было и недобавляеть условие по времени и сделать его намного короче, без подзапроса. Но я предположил, 
# что запросом будут пользоваться на постоянной основе после обновлений данных. 
# Поэтому решил все-таки оставить созданное условие по времени.


# In[49]:


# С разделением на предметы:

q_1_var_2 = '''

SELECT COUNT (st_id) AS count_id
FROM 
(
SELECT st_id, toStartOfMonth(first_time) AS first_time, 
toStartOfMonth(last_time) AS last_time,
       dateDiff('month', first_time, last_time) AS diff_time
FROM (
SELECT 
    st_id,  subject, 
    SUM ( correct ) AS sum_correct,
    MIN ( timest ) AS first_time,
    MAX ( timest ) AS last_time
FROM default.peas
GROUP BY ( st_id,  subject )
HAVING sum_correct >= 20 
)
WHERE diff_time < 1) '''

task_2_1_var_2  = ph.read_clickhouse(query=q_1_var_2, connection=connection_default)
task_2_1_var_2.count_id.item() # Выводим ответ.


# ## Задание 2.2. Образовательная платформа предлагает пройти студентам курсы по модели trial: студент может решить бесплатно лишь 30 горошин в день. Для неограниченного количества заданий в определенной дисциплине студенту необходимо приобрести полный доступ. Команда провела эксперимент, где был протестирован новый экран оплаты.
# 
# ## Необходимо в одном запросе выгрузить следующую информацию о группах пользователей:
# 
# ### ARPU 
# ### ARPAU 
# ### CR в покупку 
# ### СR активного пользователя в покупку 
# ### CR пользователя из активности по математике (subject = ’math’) в покупку курса по математике
# ### ARPU считается относительно всех пользователей, попавших в группы.
# 
# ### Активным считается пользователь, за все время решивший больше 10 задач правильно в любых дисциплинах.
# 
# ### Активным по математике считается пользователь, за все время решивший 2 или больше задач правильно по математике.

# In[50]:


q_2 = '''
SELECT (ARPAU.test_grp) AS test_grp, ARPU, ARPAU, CR, CR_ACTIVE, CR_ACTIVE_in_math
FROM (
    SELECT test_grp, 
           ROUND((income / count_id_for_study), 2) AS ARPU -- Таблица с ARPU
    FROM 
    (
        SELECT test_grp, SUM (money) AS income
        FROM
            (SELECT *
            FROM default.final_project_check) AS L_FIN
        INNER JOIN 
            (SELECT *
            FROM default.studs) R_FIN
        ON L_FIN.st_id = R_FIN. st_id 
        GROUP BY test_grp) AS L_SUM
    
    INNER JOIN 
    
        (SELECT test_grp, COUNT (DISTINCT st_id) AS count_id_for_study 
        FROM default.studs
        GROUP BY  test_grp) AS R_SUM
            ON L_SUM.test_grp = R_SUM.test_grp) AS ARPU
            
INNER JOIN 

        (
        SELECT test_grp, 
            ROUND((income / count_act_st_id), 2) AS ARPAU -- Таблица с ARPAU
        FROM
            (SELECT test_grp, SUM (money) AS income
            FROM
                (SELECT *
                FROM default.final_project_check) AS L_FIN
            INNER JOIN 
                (SELECT *
                FROM default.studs) R_FIN
            ON L_FIN.st_id = R_FIN. st_id 
            GROUP BY test_grp) AS APRAU_L
            
        INNER JOIN

            (SELECT test_grp, COUNT (DISTINCT st_id) AS count_act_st_id
            FROM
                (SELECT st_id, SUM (correct) AS correct
                FROM default.peas
                GROUP BY(st_id)
                HAVING correct > 10) AS L_cor
            
            INNER JOIN 
        
                (SELECT  st_id, test_grp 
                FROM default.studs) AS R_cor
            
            ON L_cor.st_id = R_cor.st_id
            GROUP BY test_grp) AS APRAU_R
        
        ON APRAU_L.test_grp = APRAU_R.test_grp) AS ARPAU
        
ON ARPU.test_grp = ARPAU.test_grp

INNER JOIN

        (SELECT test_grp,
               ROUND((count_id_for_final / count_id_for_study * 100), 2) AS CR -- Таблица C CR
        FROM
        (
                SELECT  test_grp,
                        COUNT (DISTINCT st_id) AS count_id_for_final
                        
                FROM 
                    (SELECT *
                    FROM default.final_project_check) AS L_CR
                    
                INNER JOIN 
                    
                    (SELECT *
                    FROM default.studs) AS R_CR
                        ON L_CR.st_id = R_CR.st_id 
                GROUP BY  test_grp) AS  CR_L
        
        INNER JOIN 
        
            (SELECT test_grp, COUNT (DISTINCT st_id) AS count_id_for_study 
            FROM default.studs
            GROUP BY  test_grp) AS CR_R
                ON CR_L.test_grp = CR_R.test_grp ) AS CR 
ON ARPAU.test_grp = CR.test_grp

INNER JOIN 

        (SELECT test_grp, ROUND((count_aciv_final / count_act_st_id * 100), 2) AS CR_ACTIVE -- -- Таблица с CR только для активных 
        FROM (
            SELECT test_grp, COUNT (DISTINCT st_id) AS count_aciv_final -- АКТИВНЫХ ИЗ FINAL
            FROM 
                    (SELECT DISTINCT st_id
                    FROM default.final_project_check
                    ) AS L_cor_ACT
            
            INNER JOIN (
                    SELECT st_id, correct, test_grp
                    FROM
                (
                    SELECT st_id, SUM (correct) AS correct
                    FROM default.peas
                    GROUP BY(st_id)
                    HAVING correct > 10) AS L_cor
                    
                    INNER JOIN 
                    
                    (SELECT  test_grp , st_id
                    FROM default.studs) AS R_cor
                    ON L_cor.st_id = R_cor.st_id) AS R_cor_ACT
                    
            ON L_cor_ACT.st_id = R_cor_ACT.st_id
            GROUP BY test_grp) AS L_ACTIV_FINAL
            
        INNER JOIN (SELECT test_grp, COUNT (DISTINCT st_id) AS count_act_st_id
                    FROM
                        (SELECT st_id, SUM (correct) AS correct
                        FROM default.peas
                        GROUP BY(st_id)
                        HAVING correct > 10) AS L_cor
                    INNER JOIN 
                        (SELECT  test_grp , st_id
                        FROM default.studs) AS R_cor
                    ON L_cor.st_id = R_cor.st_id
                    GROUP BY test_grp
                    ) AS R_ACTIV_FINAL
        ON L_ACTIV_FINAL.test_grp = R_ACTIV_FINAL.test_grp) AS CR_ACTIVE

ON CR.test_grp = CR_ACTIVE.test_grp

INNER JOIN 

        (SELECT test_grp, ROUND((count_aciv_final_in_m / count_act_st_id_in_m * 100), 2) AS CR_ACTIVE_in_math -- Таблица с CR только для активных по мтематике 
        FROM (
            SELECT test_grp, COUNT (DISTINCT st_id) AS count_aciv_final_in_m -- АКТИВНЫХ ИЗ FINAL по математике
            FROM 
                    (SELECT DISTINCT st_id
                    FROM default.final_project_check
                    WHERE subject = 'Math'
                    ) AS L_cor_ACT_M
            
            INNER JOIN (
                    SELECT st_id, correct, test_grp
                    FROM
                (
                    SELECT st_id, SUM (correct) AS correct
                    FROM default.peas
                    WHERE subject = 'Math'
                    GROUP BY(st_id)
                    HAVING correct >= 2) AS L_cor_M
                    
                    INNER JOIN 
                    
                    (SELECT  test_grp , st_id
                    FROM default.studs) AS R_cor_M
                    ON L_cor_M.st_id = R_cor_M.st_id) AS R_cor_ACT_M
                    
            ON L_cor_ACT_M.st_id = R_cor_ACT_M.st_id
            GROUP BY test_grp) AS L_ACTIV_FINAL_M
            
        INNER JOIN (SELECT test_grp, COUNT (DISTINCT st_id) AS count_act_st_id_in_m
                    FROM
                        (SELECT st_id, SUM (correct) AS correct
                        FROM default.peas
                        WHERE subject = 'Math'
                        GROUP BY(st_id)
                        HAVING correct >= 2) AS L_cor
                    INNER JOIN 
                        (SELECT  test_grp , st_id
                        FROM default.studs) AS R_cor
                    ON L_cor.st_id = R_cor.st_id
                    GROUP BY test_grp
                    ) AS R_ACTIV_FINAL_M
        ON L_ACTIV_FINAL_M.test_grp = R_ACTIV_FINAL_M.test_grp) AS CR_ACTIVE_in_math

ON CR_ACTIVE.test_grp = CR_ACTIVE_in_math.test_grp '''

task_2_2  = ph.read_clickhouse(query=q_2, connection=connection_default)
task_2_2


# # -------------------------------------------------------------------------------------------

# # Задание № 3
# 
# ## 1. Реализуйте функцию, которая будет автоматически подгружать информацию из дополнительного файла groups_add.csv (заголовки могут отличаться) и на основании дополнительных параметров пересчитывать метрики.
# 
# ## 2. Реализуйте функцию, которая будет строить графики по получаемым метрикам

# ### Выполнение задачи с помощью двух функций

# In[51]:


# Импортируем нужные библиотеки и подгрузим файл
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

group_add = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Final_project/group_add.csv')


# In[52]:


def metrics_product(group_add): # реализуем функцию, пересчитытвающую метрики
    
    # Подгружаем неободимые данные
    groups = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Final_project/groups.csv', sep = ';')
    active_studs = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Final_project/active_studs.csv')
    checks = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Final_project/checks.csv', sep = ';')
    
    # Меняем название стррок, если прислали отличные от основного датасета
    group_add.rename(columns={group_add.columns[0] : 'id', group_add.columns[1] : 'grp'})
    
    # Склеиваем groups и group_add
    group_unity = pd.concat([groups, group_add])
    
    # Проверка на корректность данных (чтобы в checks не вошли id, которых не было в active_studs, а если вошли, то добавить их в active_studs)
    active_studs_id_list = active_studs.student_id.to_list()
    checks_pass_id = checks.query('student_id != @active_studs_id_list')
    active_studs_full = pd.concat([active_studs, checks_pass_id]).drop(columns=['rev'])
  
    
    # Вмердживаем active_studs_full (уже заранее провернный датасет) в group_unity
    active_studs_full = active_studs_full.rename(columns = {'student_id' : 'id'}) # здесь просто меняем название колонки,чтобы вмерджить
    data = active_studs_full.merge(group_unity, on = 'id', how = 'left')
    
    # Вмердживаем checks в data и меняем Null на 0
    checks = checks.rename(columns = {'student_id' : 'id'})
    data_full = data.merge(checks, on = 'id', how = 'left').fillna(0)
    
    # Высчитываем метрику конверсии СR
    cr = data_full.groupby('grp', as_index = False).agg({'id' : 'count'}).rename(columns = {'id' : 'count_id'})
    cr['pay_id'] = data_full.query('rev > 0').groupby('grp', as_index = False).agg({'id' : 'count'}).id
    cr['CR'] = round((cr.pay_id / cr.count_id * 100), 2)
    cr_data = cr.drop(columns = ['count_id', 'pay_id']).reset_index(drop=True)
    
        # Высчитываем метрику Средний чек
    check_mean = data_full.query('rev > 0').groupby('grp', as_index=False)             .agg({'rev':'mean'}).rename(columns={'rev':'mean_check'})
    check_mean['mean_check'] = check_mean.mean_check.round(2)
    
    # Объединяем датасеты с метриками, чтобы выводилось все одной таблицей 
    metrics = check_mean.merge(cr_data, on = 'grp')
    
    return {'metrics' : metrics, 'df_full' :  data_full} # ставим в return еще data_full (готовый датафрэйм) для реализации доп графиков во второй функции


# In[53]:


metrics_product(group_add)['metrics'] # Проверяем, что все работает


# In[54]:


def grafics(fun): # реализуем функцию по посторению графиков метрик, рассчитанных в предыдущей фун-ции metrics_product
    
    
    #--------------------------------------- Строим графики нужных нам метрик-----------------------------------
    
    figs, axs = plt.subplots(1,2, figsize = (10,10))
    figs.tight_layout(h_pad = 2, pad = 12)
    figs.suptitle('\n                               Графики метрик', fontsize=35)
    plt.subplots_adjust(left=0,
                    bottom=0.00001,
                    right=1.5,
                    top=0.75,
                    wspace=0.4,
                    hspace=0.4)
    

    chek_bart = sns.barplot(x = 'grp', y = 'mean_check', data = fun['metrics'], ax = axs[1]).set_title('Средний чек по группам\n',
                    fontdict={'fontsize': 20,
                            'fontweight': 'bold',
                            'color': 'steelblue'}, loc='center')

    сr_bart = sns.barplot(x = 'grp', y = 'CR',  data = fun['metrics'], ax = axs[0]).set_title('CR по группам, %\n',fontdict={'fontsize': 20,
                            'fontweight': 'bold',
                            'color': 'steelblue'}, loc='center')
    
    
    # ----------------------------------------Строим дополнительные графки -----------------------------------------
    
    # График распределения среднего чека по группам
    
    fig_1, axs = plt.subplots(1,2, figsize = (10,10))
    fig_1.tight_layout(h_pad = 2, pad = 12)
    fig_1.suptitle('\n  Графики распределения среднего чека по группам', fontsize=35)
    plt.subplots_adjust(left=0,
                    bottom=0.00001,
                    right=1,
                    top=0.75,
                    wspace=0.8,
                    hspace=0.8)
    
    
    gr_A = sns.distplot(fun['df_full'].query("grp == 'A' & rev > 0").rev, ax = axs[0]).set_title('Распределение по группе А\n',
                    fontdict={'fontsize': 20,
                            'fontweight': 'bold',
                            'color': 'steelblue'}, loc='center')

    gr_B = sns.distplot(fun['df_full'].query("grp == 'B' & rev > 0").rev, ax = axs[1]).set_title('Распределение по группе В\n',fontdict={'fontsize': 20,
                            'fontweight': 'bold',
                            'color': 'steelblue'}, loc='center')
    
    # График кол-ва id в разынх группах (платящих и неплатящих)
    
    
                # Предобработаем данные
    gr_count = fun['df_full'].groupby('grp', as_index = False).agg({'id' : 'count'})
    gr_count_rev = fun['df_full'].query('rev > 0').groupby('grp', as_index = False).agg({'id' : 'count'})
    gr_count['Описание'] = 'Всего в группе'
    gr_count_rev['Описание'] = 'Всего в группе платящих'
    gr_unity=pd.concat([gr_count_rev,gr_count])
    
               # Строим график 
    fig_2, axs = plt.subplots(1,1, figsize = (10,10))
    fig_2.tight_layout(h_pad = 20, pad = 22)
    fig_2.suptitle('\n                          Кол-во id в группах', fontsize=35)
    plt.subplots_adjust(left=0.4,
                    bottom=0.001,
                    right=1,
                    top=0.75,
                    wspace=0.8,
                    hspace=0.8)

    gr_unity = sns.barplot(x = 'Описание', y = 'id', data = gr_unity, hue = 'grp').set_xlabel('')
    plt.ylabel('Количество id\n')
    
    
    
    return plt.show()


# In[55]:


grafics(metrics_product(group_add)) #Проверим, что все работает


# ### Выполнение задания с помощью одной функции 
# ### P.s. первоначально я сделал все одной функцией, а потом перечитал задание и понял, что нужно было двумя и переделал (выше представлено), ну а чтобы не пропадало зря - решил показать Вам, как сделал))))

# In[56]:


def metrics_product_and_grafics(group_add):
    
    # Подгружаем неободимые данные
    groups = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Final_project/groups.csv', sep = ';')
    active_studs = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Final_project/active_studs.csv')
    checks = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/Final_project/checks.csv', sep = ';')
    
    # Меняем название стррок, если прислали отличные от основного датасета
    group_add.rename(columns={group_add.columns[0] : 'id', group_add.columns[1] : 'grp'})
    
    # Склеиваем groups и group_add
    group_unity = pd.concat([groups, group_add])
    
    # Проверка на корректность данных (чтобы в checks не вошли id, которых не было в active_studs, а если вошли, то добавить их в active_studs)
    active_studs_id_list = active_studs.student_id.to_list()
    checks_pass_id = checks.query('student_id != @active_studs_id_list')
    active_studs_full = pd.concat([active_studs, checks_pass_id]).drop(columns=['rev'])
  
    
    # Вмердживаем active_studs_full (уже заранее провернный датасет) в group_unity
    active_studs_full = active_studs_full.rename(columns = {'student_id' : 'id'}) # здесь просто меняем название колонки, чтобы вмерджить
    data = active_studs_full.merge(group_unity, on = 'id', how = 'left')
    
    # Вмердживаем checks в data и меняем Null на 0
    checks = checks.rename(columns = {'student_id' : 'id'})
    data_full = data.merge(checks, on = 'id', how = 'left').fillna(0)
    
    # Высчитываем метрику конверсии СR
    cr = data_full.groupby('grp', as_index = False).agg({'id' : 'count'}).rename(columns = {'id' : 'count_id'})
    cr['pay_id'] = data_full.query('rev > 0').groupby('grp', as_index = False).agg({'id' : 'count'}).id
    cr['CR'] = round((cr.pay_id / cr.count_id * 100), 2)
    cr_data = cr.drop(columns = ['count_id', 'pay_id']).reset_index(drop=True)
    
        # Высчитываем метрику Средний чек
    check_mean = data_full.query('rev > 0').groupby('grp', as_index=False)             .agg({'rev':'mean'}).rename(columns={'rev':'mean_check'})
    check_mean['mean_check'] = check_mean.mean_check.round(2)
    
    # Выведем на экран таблицы с метриками
    
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(.5)

    
    cr = fig.add_subplot(2,2,1)
    cr.table(cellText = cr_data.values, 
         colLabels = cr_data.columns, cellLoc='center')
    cr.set_title('Конверсия CR', 
              fontdict={'fontsize': 20,
                        'fontweight': 'bold',
                        'color': 'steelblue'},
              loc='center')    
    cr.axis('off')
    

    check = fig.add_subplot(2,2,2)
    check.table(cellText = check_mean.values, 
         colLabels = check_mean.columns, cellLoc='center')  
    check.set_title('Средний чек', 
              fontdict={'fontsize': 20,
                        'fontweight': 'bold',
                        'color': 'steelblue'},
              loc='center')
    check.axis('off')
    

    #----------------------------------------------------Графики--------------------------------------------------------------
    
    # -------------График метрик-------------
    
    figs, axs = plt.subplots(1,2, figsize = (10,10))
    figs.tight_layout(h_pad = 2, pad = 12)
    figs.suptitle('\n                               Графики метрик', fontsize=35)
    plt.subplots_adjust(left=0,
                    bottom=0.00001,
                    right=1.5,
                    top=0.75,
                    wspace=0.4,
                    hspace=0.4)
    

    chek_bart = sns.barplot(x = 'grp', y = 'mean_check', data = check_mean, ax = axs[1]).set_title('Средний чек по группам\n',
                    fontdict={'fontsize': 20,
                            'fontweight': 'bold',
                            'color': 'steelblue'}, loc='center')

    сr_bart = sns.barplot(x = 'grp', y = 'CR',  data = cr_data, ax = axs[0]).set_title('CR по группам, %\n',fontdict={'fontsize': 20,
                            'fontweight': 'bold',
                            'color': 'steelblue'}, loc='center')
    
    # --------Дополнительные графики (часть 1 - Распределения)-----------
    
    fig_1, axs = plt.subplots(1,2, figsize = (10,10))
    fig_1.tight_layout(h_pad = 2, pad = 12)
    fig_1.suptitle('\n  Графики распределения среднего чека по группам', fontsize=35)
    plt.subplots_adjust(left=0,
                    bottom=0.00001,
                    right=1,
                    top=0.75,
                    wspace=0.8,
                    hspace=0.8)
    

    gr_A = sns.distplot(data_full.query("grp == 'A' & rev > 0").rev, ax = axs[0]).set_title('Распределение по группе А\n',
                    fontdict={'fontsize': 20,
                            'fontweight': 'bold',
                            'color': 'steelblue'}, loc='center')

    gr_B = sns.distplot(data_full.query("grp == 'B' & rev > 0").rev, ax = axs[1]).set_title('Распределение по группе В\n',fontdict={'fontsize': 20,
                            'fontweight': 'bold',
                            'color': 'steelblue'}, loc='center')
    
    
    # ---------Дополнительные графики (часть 2 - Кол-во id)-----------------
    
            # Предобработаем данные
    gr_count = data_full.groupby('grp', as_index = False).agg({'id' : 'count'})
    gr_count_rev = data_full.query('rev > 0').groupby('grp', as_index = False).agg({'id' : 'count'})
    gr_count['Описание'] = 'Всего в группе'
    gr_count_rev['Описание'] = 'Всего в группе платящих'
    gr_unity=pd.concat([gr_count_rev,gr_count])
    
            # Строим график 
    fig_2, axs = plt.subplots(1,1, figsize = (10,10))
    fig_2.tight_layout(h_pad = 20, pad = 22)
    fig_2.suptitle('\n                          Кол-во id в группах', fontsize=35)
    plt.subplots_adjust(left=0.4,
                    bottom=0.001,
                    right=1,
                    top=0.75,
                    wspace=0.8,
                    hspace=0.8)

    gr_unity = sns.barplot(x = 'Описание', y = 'id', data = gr_unity, hue = 'grp').set_xlabel('')
    plt.ylabel('Количество id\n')

    return cr, check, plt.show()


# In[57]:


metrics_product_and_grafics(group_add) # Проверяем, что все работает


# In[ ]:




