#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
olist_customers_datase = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/For_project/Var_1/olist_customers_dataset.csv')
olist_orders_dataset = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/For_project/Var_1/olist_orders_dataset.csv')
olist_order_items_dataset = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-v-mazur/For_project/Var_1/olist_order_items_dataset.csv')


# In[2]:


olist_customers_datase


# In[3]:


olist_orders_dataset


# In[4]:


olist_order_items_dataset


# In[5]:


# Покупка своерешена, если order_status не отменен или не недоступен. Считаю, что по факту оплаты нельзя смотреть, потому что многие платят, а потом отказыватся (это видно из этой таблицы), как и по времени доставки (заказан отменен, но доставка все-равно была). А бывает, что пишет, что доставка есть, а времени нет (т.е. сам забрал - 8 таких наблюдений)
buy = olist_orders_dataset.query("order_status !='canceled' & order_status !='unavailable'")#.order_delivered_customer_date.isna().sum()


# In[6]:


# Также нужно убрать тех, кто не отплатил order_approved_at - пустое
buy = buy.dropna(subset=['order_approved_at'])


# In[7]:


buy # итог того, что мы считаем покупкой


# In[8]:


# задание 1
buy_customer_id = olist_customers_datase.merge(buy, on = 'customer_id') # сведем с таблицей, показывающей уникальный идентификатор пользователя 


# In[9]:


buy_customer_id


# In[10]:


buy_customer_id_one_orders = buy_customer_id.groupby('customer_unique_id', as_index = False).agg({'customer_id' : 'count'}).rename(columns = {'customer_id' : 'number_orders'}).query('number_orders == 1')
# найдем сколько пользователей с одним заказом (уже отобранных, по которым совершена предполагаемая покупка)


# In[11]:


buy_customer_id_one_orders.customer_unique_id.nunique() # вывдем ответ


# In[12]:


# задание 2
olist_orders_dataset


# In[13]:


not_delivery = olist_orders_dataset[olist_orders_dataset['order_delivered_customer_date'].isnull()] # вывели тех, у кого не было доставки


# In[14]:


not_delivery


# In[15]:


not_delivery['order_estimated_delivery_date'] = pd.to_datetime(not_delivery['order_estimated_delivery_date']).dt.month # приведем все к месяцу


# In[16]:


not_delivery


# In[17]:


not_delivery_month = not_delivery.groupby('order_estimated_delivery_date', as_index = False).agg({'order_id' : 'count'}).rename(columns = {'order_id' : 'count_not_del'}) # сколько в приципе не доставляется по месяцам


# In[18]:


not_delivery_month


# In[19]:


mean_not_del = not_delivery_month.count_not_del.mean() 


# In[20]:


mean_not_del # в среднем не доставялется 247 заказаов


# In[21]:


# Посмотрим на причины


# In[22]:


not_delivery.order_status.value_counts() # посмотрим со стороны статуса (потому что отменен или недостпен в сумме составялет 619 + 609 = 1228 из 2965). Наибольшее число составляет по "отгружен со склада".Проверим по shipped 


# In[ ]:





# In[23]:


not_delivery.query("order_status == 'shipped'").groupby('order_estimated_delivery_date', as_index = False).agg({'order_id' : 'count'}).sort_values('order_id') # видно, что по shiped наибольшее число недоеханный - 5 месяц. Далее, посмотрим, сколько число заказов в этих месяцах в принципе. Может, мы просто не справляемся с большим чслом заказов... Или имеется ввиду, что shiperd - это забран со скалада?? Сейчас мы проверим...


# In[24]:


olist_orders_dataset['order_estimated_delivery_date'] = pd.to_datetime(olist_orders_dataset['order_estimated_delivery_date']).dt.month


# In[25]:


olist_orders_dataset.query("order_status == 'shipped'").groupby('order_estimated_delivery_date', as_index = False).agg({'order_id' : 'count'}).rename(columns = {'order_id' : 'count_del'}).sort_values('count_del') # Значения по shipered в принципе по всем заказам совпадает с значениями, по которым стоит "пусто" из таблицы not_delivery 


# In[26]:


olist_orders_dataset.query("order_status == 'shipped'").order_delivered_customer_date.isna().sum() # докозательство, что shipered - это отгружен со склада и не должен иметь в принципе времени со склада (1107 всего (вывели данным кодом) и 1107 с пустыми данными о времени доставки (вывели в not_delivery))


# In[27]:


# Разобрались с shipered (там все ясно, времени и недолжно быть), но не понятно, почему по delivered есть пустые, посмотрим на них внимательно (по остальным также все понятно: то в процесее, то только выставлен счет и тд).


# In[28]:


not_delivery_month_delivered = not_delivery.query("order_status == 'delivered'").groupby('order_estimated_delivery_date', as_index = False).agg({'order_id' : 'count'}).rename(columns = {'order_id' : 'count_not_del'}) # сколько в приципе не доставляется по месяцам


# In[29]:


not_delivery_month_delivered


# In[30]:


not_delivery_month_delivered.count_not_del.mean() # выведем среднее только для них


# In[31]:


# Посмотрим сколько в этих месяцах было заказов (может, не справлялись)
olist_orders_dataset.query("order_status == 'delivered'").groupby('order_estimated_delivery_date', as_index = False).agg({'order_id' : 'count'}).rename(columns = {'order_id' : 'count_del'}).sort_values('count_del')
# видно, что зависело не от этого. Самое большое кол-во заказов в августе, но там все было хорошо с доставкой


# In[32]:


#Глянем в таком случае на штаты
not_delivery_delivered = not_delivery.query("order_status == 'delivered'")


# In[33]:


not_delivery_delivered


# In[34]:


merge = olist_customers_datase.merge(not_delivery_delivered, on = 'customer_id')


# In[35]:


merge #почему-то заказы не доходят в штат SP (при этом, в логистическую компанию заказ был передан). А в RS не дошло, потому что в принципе не передали в логистическую компанию


# In[36]:


# задание 3 По каждому товару определить, в какой день недели товар чаще всего покупается.
olist_order_items_dataset#.product_id.nunique()


# In[37]:


buy_product_id = olist_order_items_dataset.merge(buy, on = 'order_id') # соединим с тем, что считаем покупкой и товарные позиции, входящие в заказы


# In[38]:


buy_product_id['order_approved_at'] = pd.to_datetime(buy_product_id['order_approved_at']).dt.day_name()


# In[39]:


buy_product_id


# In[40]:


buy_product_id_week = buy_product_id.groupby(['product_id', 'order_approved_at'], as_index = False).agg({'order_id' : 'count'}).rename(columns = {'order_id' : 'prod_count'}).sort_values('prod_count', ascending = False).drop_duplicates(subset='product_id')


# In[41]:


buy_product_id_week


# In[42]:


# Ответ: вторник


# In[43]:


# задание 4
#Сколько у каждого из пользователей в среднем покупок в неделю (по месяцам)? Не стоит забывать, что внутри месяца может быть не целое количество недель. Например, в ноябре 2021 года 4,28 недели. И внутри метрики это нужно учесть.


# In[44]:


buy_customer_id


# In[45]:


# проверим какие года есть (есть ли среди них високосные). 2016 - висосоксный есть, но по нему нет февраля, значит функция, представленная ниже def week_order_approved_at (order_approved_at) по 2 месяцу не имеет ошибки
year = buy_customer_id.copy()
year['order_approved_at'] = pd.to_datetime(year['order_approved_at']).dt.strftime('%Y-%m')
print(year.order_approved_at.unique())


# In[46]:


purchase = buy_customer_id.groupby(['customer_unique_id', 'order_approved_at'], as_index = False).agg({'customer_id' : 'count'}).rename (columns = {'customer_id' : 'purchase'}).sort_values('purchase', ascending = False) # выели число покупок кажого клиента в каждый месяц 


# In[47]:


purchase['order_approved_at'] = pd.to_datetime(purchase['order_approved_at']).dt.month # перевдем в месяц даты покупок


# In[48]:


purchase


# In[49]:


# Создадим функцию, которая высчитвает сколько дней в месяце
def week_order_approved_at (order_approved_at):
    if (order_approved_at == 4) or (order_approved_at == 6) or (order_approved_at == 9) or (order_approved_at == 11):
        day = 30
    elif order_approved_at == 2:
        day = 28
    else:
        day = 31
    return day


# In[50]:


purchase['day_in_month'] = purchase.order_approved_at.apply(week_order_approved_at) # создаем колонку с днями в каждом месяце


# In[51]:


# посчитаем кол-во недель, разделив количесвто дней на 7 (дней в каждую неделю)
purchase['week_in_month'] = purchase.day_in_month / 7


# In[52]:


purchase


# In[53]:


# теперь выведем сколько у каждого из пользователей в среднем покупок в неделю (по месяцам)?
purchase['purchase_mean_week_in_month'] = purchase.purchase / purchase.week_in_month


# In[54]:


purchase


# In[55]:


# задание 5
buy_customer_id.info() # за базу снова берем то, что считаем покупкой 


# In[56]:


#buy_customer_id_no_one = buy_customer_id.groupby(['customer_unique_id', 'order_approved_at'], as_index = False)\
#.agg({'customer_id' : 'count'})\
#.rename(columns = {'customer_id' : 'number_orders'})\
#.query('number_orders != 1') - 'n'
    
    
#cohort = buy_customer_id_no_one
#buy_customer_id_no_one['order_approved_at'] = pd.to_datetime(buy_customer_id_no_one['order_approved_at']) # перевдем колонку во время
#buy_customer_id_no_one["min_time"] = buy_customer_id_no_one.groupby('customer_unique_id')['order_approved_at'].transform('min')
#buy_customer_id_no_one.query('order_approved_at != min_time') - это я делал для себя, чтобы понять, что transform('min') работает как мне надо


# In[57]:


# buy_customer_id_no_one.query('order_approved_at != min_time') - проверка та же


# In[58]:


cohort = buy_customer_id.copy()
cohort['order_approved_at'] = pd.to_datetime(cohort['order_approved_at']).dt.strftime('%Y-%m')


# In[59]:


cohort['min_time'] = cohort.groupby('customer_unique_id')['order_approved_at'].transform('min') # - выводим колонку с первой покупкой


# In[60]:


cohort.info()


# In[61]:


# ранее, когда мы проверяли даты на високосный год, можно было заметить, что целый год, за который у нас есть данные, является только 2017 год (2016 год и 2018 - не полные)
# Значит, будет проводить анализ только на базе 2017 года (т.к спрашивают за январь - декабрь, т.е. целый год): 
cohort_2017 = cohort.query("min_time >= '2017-01' & min_time <= '2017-12'")


# In[62]:


cohort_2017#.query('order_approved_at != min_time') #- это можно использовать, чтобы проверить, что min_time и order_approved_at отличаются


# In[63]:


cohort_2017_uniq_id = cohort_2017.groupby(['order_approved_at' , 'min_time'], as_index = False).agg({'customer_unique_id' : 'nunique'}) # - группируем по временам и выводим уникальных покупателей


# In[64]:


cohort_2017_uniq_id['order_approved_at'] = pd.to_datetime(cohort_2017_uniq_id['order_approved_at'])
cohort_2017_uniq_id['min_time'] = pd.to_datetime(cohort_2017_uniq_id['min_time'])


# In[65]:


cohort_2017_uniq_id.info()


# In[66]:


cohort_2017_uniq_id ['diff_time'] = cohort_2017_uniq_id.order_approved_at.dt.to_period('M').astype(int) - cohort_2017_uniq_id.min_time.dt.to_period('M').astype(int)


# In[67]:


cohort_2017_uniq_id


# In[68]:


cohort_2017_uniq_id_pivot = cohort_2017_uniq_id.pivot(index = 'min_time', columns = 'diff_time', values =  'customer_unique_id')


# In[69]:


cohort_2017_uniq_id_pivot # - таким обраозом, мы определели сколько уникальных пользователей возвращались через 0-19 месяцев (начиная с января 2017 года)


# In[70]:


cohort_2017_uniq_id_pivot['retention_in_3_month'] = round(((cohort_2017_uniq_id_pivot[3] / cohort_2017_uniq_id_pivot[0]))*100,2) # находим отношение пользователей которые вернулись на 3 день к чсилу пользоватей в первый месяц (01-01-2017)


# In[71]:


cohort_2017_uniq_id_pivot


# In[72]:


cohort_2017_uniq_id_pivot.retention_in_3_month.idxmax()


# In[73]:


# 6 задание
# Для начала предобработаем olist_order_items_dataset
olist_order_items_dataset_price = olist_order_items_dataset.groupby('order_id', as_index = False).agg({'price' : 'sum'})


# In[74]:


# теперь объединим таблицы buy_customer_id - где, как мы считаем,находятся клиенты, которые действительно покупали, с olist_order_items_dataset_price, где указана сумма по каждому заказу
segment = buy_customer_id.merge(olist_order_items_dataset_price, on = 'order_id')


# In[75]:


segment['order_approved_at'] = pd.to_datetime(segment['order_approved_at'])


# In[76]:


segment


# In[77]:


# Допустим, мы смотрим ежедневыне отчеты (т.е. отчетный период смотрится на день позже фактического дня). Определим последний день покупки из представленных наших данных ('это 3 сентября 2018')
last_month = segment.order_approved_at.max()
print(last_month)


# In[78]:


# Определим отчетный день (4 сентября 2018 года) 
from datetime import timedelta
today = last_month + timedelta(days=1)
print(today)


# In[79]:


# создаим таблицу, отражающкю RMF показатель: 
# R - время от последней покупки пользователя до текущей даты
# F - суммарное количество покупок у пользователя за всё время
# M - сумма покупок за всё время


# In[80]:


rfm_segment = segment.groupby('customer_unique_id', as_index = False).agg({'order_approved_at' : lambda x: (today - x.max()).days,'customer_id' : 'count', 'price' : 'sum'}).rename(columns = {'order_approved_at' : 'Recency','customer_id' : 'Frequency', 'price' : 'Monetary'}) # - добавили колонки R, F и M


# In[98]:


rfm_segment.Frequency.value_counts()


# In[82]:


# Определим, что разбивать будем на 5 частей, т.е. брать 3 квантиля
rfm_segment_quantile = rfm_segment[['Recency', 'Frequency', 'Monetary']].quantile(q=[0.2, 0.4, 0.6, 0.8]).to_dict()


# In[83]:


rfm_segment_quantile


# In[84]:


# Создадим функции, отражающие показатели наших разбивок (побальная система)
# 1-я функция будет использована для R: чем недавнее клиент заходил, тем выше его оценка
def r_score(x):
    if x <= rfm_segment_quantile['Recency'][0.2]:
        return 5
    elif x <= rfm_segment_quantile['Recency'][0.4]:
        return 4
    elif x <= rfm_segment_quantile['Recency'][0.6]:
        return 3
    elif x <= rfm_segment_quantile['Recency'][0.8]:
        return 2
    else:
        return 1
# 2 функция будет использована для F и M: чем меньше сумма покупок и коли-во покупок, тем ниже оценка
def fm_score(x, c):
    if x <= rfm_segment_quantile[c][0.2]:
        return 1
    elif x <= rfm_segment_quantile[c][0.4]:
        return 2
    elif x <= rfm_segment_quantile[c][0.6]:
        return 3
    elif x <= rfm_segment_quantile[c][0.8]:
        return 4
    else:
        return 5 


# In[85]:


# Применяем функцию к нашим полученным данным rfm_segment
rfm_segment['R'] = rfm_segment['Recency'].apply(lambda x: r_score(x))
rfm_segment['F'] = rfm_segment['Frequency'].apply(lambda x: fm_score(x, 'Frequency'))
rfm_segment['M'] = rfm_segment['Monetary'].apply(lambda x: fm_score(x, 'Monetary'))


# In[86]:


rfm_segment


# In[87]:


# Теперь образуем RFM - данные
rfm_segment['RFM'] = rfm_segment['R'].map(str) + rfm_segment['F'].map(str) + rfm_segment['M'].map(str)


# In[88]:


rfm_segment['RFM'] = rfm_segment['RFM'].astype(int) # переведем колонку в int


# In[89]:


rfm_segment = rfm_segment.sort_values('RFM') # произведем сортировку для удобства


# In[90]:


rfm_segment


# In[91]:


rfm_segment.RFM.unique() # - выведем все возможные


# In[92]:


# Разделим, согласно полученным данным:
#111-115 / 211-215 / 311-315 - "Одноразовые" покуатели (покупали давно и немного раз)
#151-155 / 251-255 / 351-355 - "Ушедшие" покупатели (покупали много или за большие цены, но ушли от нас)
#411-415  - "Недавние" покупатели (совершили покупки относительно недавно, но единожды)
#451-455 / 551-555 - "Потенциально - постоянные" покупатели (соверешили много покупок и относительно недавно)
#511-515 - "Новенькие" покупатели (приишли недавно и совершили мало покупок)


# In[135]:


def anal(x):
    if 111 <= x <= 115 or 211 <= x <= 215 or 311 <= x <= 315:
        return ('Одноразовыe')
    elif 151 <= x <= 155 or 251 <= x <= 255 or 351 <= x <= 355:
        return ('Ушедшие')
    elif 411 <= x <= 415:
        return ('Недавние')
    elif 451 <= x <= 455 or 551 <= x <= 555:
        return ('Потенциально - постоянные')
    elif 511 <= x <= 515:
        return("Новенькие")


# In[136]:


rfm_segment['name_rfm'] = rfm_segment['RFM'].apply(anal)


# In[139]:


rfm_segment.name_rfm.unique()


# In[144]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (11,8))
sns.countplot(rfm_segment['name_rfm'])


# In[ ]:




