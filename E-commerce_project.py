# Проект e-commerce: 
# Описание данных:

#  olist_customers_datase.csv — таблица с уникальными идентификаторами пользователей
# customer_id — позаказный идентификатор пользователя
# customer_unique_id —  уникальный идентификатор пользователя  (аналог номера паспорта)
# customer_zip_code_prefix —  почтовый индекс пользователя
# customer_city —  город доставки пользователя
# customer_state —  штат доставки пользователя

# olist_orders_dataset.csv —  таблица заказов
# order_id —  уникальный идентификатор заказа (номер чека)
# customer_id —  позаказный идентификатор пользователя
# order_status —  статус заказа
# order_purchase_timestamp —  время создания заказа
# order_approved_at —  время подтверждения оплаты заказа
# order_delivered_carrier_date —  время передачи заказа в логистическую службу
# order_delivered_customer_date —  время доставки заказа
# order_estimated_delivery_date —  обещанная дата доставки

# olist_order_items_dataset.csv —  товарные позиции, входящие в заказы
# order_id —  уникальный идентификатор заказа (номер чека)
# order_item_id —  идентификатор товара внутри одного заказа
# product_id —  ид товара (аналог штрихкода)
# seller_id — ид производителя товара
# shipping_limit_date —  максимальная дата доставки продавцом для передачи заказа партнеру по логистике
# price —  цена за единицу товара
# freight_value —  вес товара

# Уникальные статусы заказов в таблице olist_orders_dataset:
# created —  создан
# approved —  подтверждён
# invoiced —  выставлен счёт
# processing —  в процессе сборки заказа
# shipped —  отгружен со склада
# delivered —  доставлен пользователю
# unavailable —  недоступен
# canceled —  отменён

import pandas as pd
import numpy as np
import calendar as cl
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')

# загрузка данных
customers_data=pd.read_csv('olist_customers_dataset.csv')
orders_data=pd.read_csv('olist_orders_dataset.csv', parse_dates=['order_purchase_timestamp', 'order_approved_at','order_delivered_carrier_date',                                      'order_delivered_customer_date', 'order_estimated_delivery_date'])
order_items_data=pd.read_csv('olist_order_items_dataset.csv', parse_dates=['shipping_limit_date'])

# Проверка данных на дубликаты, пропущенные значения, соответствие типов данных
customers_data.isna().sum() # пропущенных значений нет
customers_data.dtypes # типы соответствуют
customers_data.loc[customers_data.duplicated()] # дубликатов нет
orders_data.isna().sum() # пропущенные значения показывают, что не все заказы оплачены/переданы в логистику/доставлены. Оставляем их.
orders_data.dtypes # колонки с датами обработаны при загрузке данных
orders_data.loc[orders_data.duplicated()] # дубликатов нет
order_items_data.isna().sum() # пропущенных значений нет
order_items_data.dtypes # колонка с датой обработана при загрузке данных
order_items_data.loc[order_items_data.duplicated()] # дубликатов нет

# 1. Определить кол-во пользователей, которые совершили покупку только один раз?

# Что считать покупкой?
# Покупкой считается момент перехода права собственности на товар. Так как проект e-commerce, то право собственности на товар
# (при продаже товара через интернет-магазин с условием доставке товара покупателю) переходит к покупателю в момент получения
# товара непосредственно покупателем (ОБОСНОВАНИЕ: Согласно ч.1 ст. 223 ГК РФ право собственности у приобретателя вещи по
# договору возникает с момента ее передачи, если иное не предусмотрено законом или договором.)
# Таким образом, будем считать покупку совершенной, если есть факт доставки (время доставки заказа не пустое).

# Посмотрим на заказы, доставленные покупателям. Для этого объединим таблицы пользователей и заказов по полю 'customer_id' и 
# отберем строки, в которых есть время доставки заказа.

# объединение таблиц
merge_df=customers_data.merge(orders_data, how='inner', on='customer_id')
# фильтрация и группировка данных 
delivered=merge_df.query('order_delivered_customer_date != "NaT"')
delivered.groupby('order_status', as_index=False).agg({'customer_id':'count'})

# Доставленные заказы имеют статусы 'доставлен' и 'отменен'. Статус canceled: судя по данным заказ доставлен и сразу отменен
delivered.groupby('customer_unique_id', as_index=False).agg({'customer_id':'count'}).query('customer_id == 1').count()

# 1. Ответ: 90555 пользователей совершили покупку только один раз

# 2. Сколько заказов в месяц в среднем не доставляется по разным причинам (вывести детализацию по причинам)?
# Посмотрим на заказы, в которых отсутствует дата доставки заказа
merge_df.query('order_delivered_customer_date == "NaT"').groupby('order_status', as_index=False).agg({'customer_id':'count'})

# Заказы со статусами created, approved, invoiced, processing, shipped - и не должны иметь время доставки, поэтому их не
# рассматриваем. Интересуют статусы: delivered, canceled, unavailable. Рассмотрим их.

# Статус delivered
merge_df.query('order_delivered_customer_date == "NaT" and order_status=="delivered"').head()   
# Обещанные даты доставки больше дат передачи заказов в службу логистики (у них есть все шансы доехать до своих покупателей),
# Рассматривать эти заказы как НЕ доставленные НЕ будем

# Статус canceled. Это заказы, отмененные на разных этапах. Сохраним их в not_delivered_1
not_delivered_1=merge_df.query('order_delivered_customer_date == "NaT" and order_status=="canceled"')
not_delivered_1.head(3)

# Статус unavailable
merge_df.query('order_delivered_customer_date == "NaT" and order_status=="unavailable"').head(3)

# Чтобы принять решение по заказам 'unavailable' посмотрим на максимальную дату доставки продавцом для передачи в службу
# доставки (shipping_limit_date), то есть определим есть ли шанс у заказа быть доставленным в срок (обещанный срок доставки).

# Для этого объединим данные с таблицей по товарным позициям по ключу order_id 
merge_df_total=merge_df.merge(order_items_data, how='left', on='order_id')

# У этих заказов не обозначен срок поставки товаров от поставщика, нам не известно в какие сроки они поступят на склад, поэтому
# будем рассматривать их как НЕ доставленные. Сохраним их в not_delivered_2
not_delivered_2=merge_df_total.query('order_delivered_customer_date == "NaT" and order_status=="unavailable" and shipping_limit_date=="NaT"')

# У этих заказов 'unavailable' обозначена дата поставки товаров от поставщика и она меньше обещанной даты доставки заказов. Предполагаем, что
# товары будут доставлены в срок (дадим им этот шанс), поэтому НЕ будем рассматривать их как НЕ доставленные

merge_df_total.query('order_delivered_customer_date == "NaT" and order_status=="unavailable" and shipping_limit_date <= order_estimated_delivery_date ').head(3)

# объединение данных по недоставленным товарам
not_delivered=pd.concat([not_delivered_1, not_delivered_2]) 

# создание колонки обещанной даты доставки в формате месяц-год для дальнейшего расчета среднего кол-ва недоставленных товаров
# по месяцам
not_delivered['month_estimated']=not_delivered['order_estimated_delivery_date'].dt.strftime('%Y-%m') 
not_delivered_by_status=not_delivered.groupby(['order_status', 'month_estimated'], as_index=False).agg({'order_id':'count'}).groupby('order_status', as_index=False)                                      .agg({'order_id':'mean'})                                      .rename(columns={'order_id':'mean_not_delivered'})
not_delivered_by_status['mean_not_delivered']=not_delivered_by_status.mean_not_delivered.round(0)
not_delivered_by_status

# 2. Ответ: по причине отмены в среднем не доставляется 24 заказа в месяц, по причине недоступности товаров в среднем не
# доставляется 30 заказов в месяц


# 3. По каждому товару определить, в какой день недели товар чаще всего покупается (используется дата создания заказа (order_purchase_timestamp)
# скопируем датафрейм, чтобы не вносить изменения в первоначальные данные
merge_df_total_copy=merge_df_total.copy(deep=True)

# создание колонки с днем недели заказа
merge_df_total_copy['weekday']=merge_df_total_copy['order_purchase_timestamp'].dt.strftime('%A') 

# подсчет кол-ва заказов по товару и дням недели
df_items=merge_df_total_copy.groupby(['product_id', 'weekday'], as_index=False).agg({'seller_id': 'count'})  
df_items=df_items.rename(columns={'seller_id':'count_orders'})

# создание датафрейма с уникальными id товара и максимальним значением покупки в день
df_max=df_items.groupby('product_id', as_index=False).agg({'count_orders':'max'}) 

# объединение данных по id товара и максимальному значению. Так не потеряются товары, у которых существует два и более дней
# недели с максимальным колличеством заказов
df_result=df_items.merge(df_max, how='inner', on=['product_id', 'count_orders'])

# Ответ 3: итоговая таблица
df_result.head()

df_result.product_id.nunique() # для проверки посмотрим кол-во уникальных id товаров в первоначальной таблице и в полученной
merge_df_total_copy.product_id.nunique()

# пример товара с двумя днями недели, когда товар чаще покупался
df_result.query('product_id == "002159fe700ed3521f46cfcf6e941c76"') 

# 4. Сколько у каждого из пользователей в среднем покупок в неделю (по месяцам)? Не стоит забывать, что внутри месяца может быть
# не целое количество недель. Например, в ноябре 2021 года 4,28 недели. И внутри метрики это нужно учесть.

# обновим копию датафрейма
merge_df_total_copy=merge_df_total.copy(deep=True)

# фильтр по непустым датам доставки
merge_df_total_copy=merge_df_total_copy.query('order_delivered_customer_date != "NaT"')
# создание колонки с месяцем покупки 
merge_df_total_copy['delivered_month']=merge_df_total_copy['order_delivered_customer_date'].dt.strftime('%B')
# создание колонки с колличеством недель в месяце 
merge_df_total_copy['week_in_month']=merge_df_total_copy.order_delivered_customer_date.dt.daysinmonth/7
merge_df_total_copy.head(3)

# Посчитаем кол-во покупок по каждому покупателю и каждому заказу для пользователя. В этой группировке у одного покупателя
# может быть два одинаковых месяца, т.к. они из разных годов покупок
purchase_df_month=merge_df_total_copy.groupby(['customer_unique_id', 'order_id', 'delivered_month', 'week_in_month'], as_index=False).agg({'order_item_id':'count'}).rename(columns={'order_item_id':'count_purchase'})

# Подсчитаем колличество покупок в неделю (в каждом месяце в каждый год покупок) путем деления кол-ва покупок в месяц на кол-во 
# недель в месяце
purchase_df_month['count_in_week']=purchase_df_month.count_purchase/purchase_df_month.week_in_month

# Сгруппируем по месяцам и подсчитаем среднее колличество покупок в неделю
otvet=purchase_df_month.groupby(['customer_unique_id', 'delivered_month'], as_index=False).agg({'count_in_week':'mean'})

# Итоговая таблица к заданию 4
otvet[['customer_unique_id', 'delivered_month', 'count_in_week']].head()

# пример по пользователю с большим колличеством покупок
otvet.query('customer_unique_id=="8d50f5eadf50201ccdcedfb9e2ac8455"')

# 5.Провести когортный анализ пользователей. В период с января по декабрь выявить когорту с самым высоким retention на 3й месяц.

merge_df_total_copy=merge_df_total.copy(deep=True)

# фильтр по непустым датам доставки
merge_df_total_copy=merge_df_total_copy.query('order_delivered_customer_date != "NaT"')

# создание колонки с датой покупки в формате год-месяц 
merge_df_total_copy['orders_period']=merge_df_total_copy['order_delivered_customer_date'].dt.strftime('%Y-%m')

# посмотрим как распределяются покупки пользователей по месяцам
by_month=merge_df_total_copy.groupby(['orders_period'], as_index=False).agg({'customer_unique_id':'count'}).sort_values('orders_period').rename(columns={'customer_unique_id': 'count_orders'})
plt.figure(figsize=(15,6))
plt.xticks(np.arange(len(by_month)), by_month['orders_period'], rotation=90)
sns.lineplot(x='orders_period', y ='count_orders', data=by_month)

# исходя из результатов возьмем 2017 год для формирования когорт, плюсом захватим 3 месяца 2018 года, чтобы отследить 
# retention на 3й месяц для когорты декабря 2017 

# определим для каждого покупателя дату его первой покупки

first_orders =merge_df_total_copy.groupby('customer_unique_id', as_index=False).agg({'order_delivered_customer_date': 'min'}).rename(columns={'order_delivered_customer_date':'first_orders_period'})
first_orders['first_orders_period']=first_orders['first_orders_period'].dt.strftime('%Y-%m')

# добавим колонку с датой первой покупки к общей таблице и возьмем нужные нам колонки
new_merge = merge_df_total_copy.merge(first_orders, how='inner', on='customer_unique_id')
new_merge = new_merge[['customer_unique_id', 'orders_period', 'first_orders_period']]

# определим общее кол-во пользователей, совершивших покупку по месяцам и добавим этот столбец к таблице с предыдущего шага
total_user=new_merge.groupby('first_orders_period', as_index=False).agg({'customer_unique_id':'nunique'}).rename(columns={'customer_unique_id':'total_user'})
df_cohorts=new_merge.merge(total_user, how='left', on='first_orders_period')

# проведем группировку данных: верхняя группировка по first_orders_period (когорты), следом по orders_period (периоды когорт).
# Считаем кол-во пользователей из когорты, совершивших покупку

cohorts=df_cohorts.groupby(['first_orders_period', 'orders_period']).agg({'customer_unique_id':'nunique', 'total_user':'max'}).rename(columns={'customer_unique_id': 'users_of_cohorts'})

# будем рассматривать когорты только за 2017 год плюс 3 месяца 2018, чтобы отследить CRR на третий месяц
cohorts=cohorts.query('"2016-12" < first_orders_period < "2018-01" and orders_period < "2018-04"')
# переиндексируем датафрейм
cohorts.reset_index(inplace=True)

# формула retention rate: CRR =
# ((Кол-во клиентов на конец периода (total_user) — Новые пользователи) / Кол-во клиентов в когорте(users_of_cohorts)) * 100%
# новые пользователи = все пользователи - Кол-во пользователей из когорты (совершившие повторно покупку)
cohorts['CRR']=round(cohorts.users_of_cohorts/cohorts.total_user*100, 2)
cohorts.head()

# Добавим порядковое значение периода когорты (месяца, когда мы ждем повторной покупки) для каждой когорты
def cohort_period(date):
    period=0
    sp=date.split('-')
    if sp[0]=="2017":
        period=int(sp[1])
    else:
        period=int(sp[1])+12
    return(period)

cohorts['cohort_period'] = cohorts.orders_period.apply(cohort_period)

# вернем обратно индексы, но теперь с когортами и периодами когорт, вынесем второй уровень индексов в столбцы
cohorts.set_index(['first_orders_period', 'cohort_period'], inplace=True)
table=cohorts['CRR'].unstack(1)

# итоговая таблица с CRR по когортам и периодам
table

# Тепловая карта.
# Как видим, CRR для всех когорт имеет не особо высокие значения, и самые большие из них наблюдаются в первый месяц после месяца
# формирования когорт. Таким образом, мы можем судить о не способности бизнеса удерживать своих покупателей. В данном примере 
# видно, что менее 1% пользователей совершают повторные покупки 

cmap = sns.cm.rocket_r
sns.set(style='ticks')
plt.figure(figsize=(15, 9))
plt.xlabel('Cohorts')
plt.title('Cohorts: Retention Rate')
sns.heatmap(table, annot=True, cmap=cmap, vmin=0, vmax=2, center= 1)

# найдем когорту с самым большим CRR на третий месяц
max_crr=cohorts.copy()
max_crr.reset_index(inplace=True)

sp1, sp2=[], []
for i, row in max_crr.iterrows():
    coh=int(row['first_orders_period'].split('-')[1])
    if row['cohort_period']==(coh+3):
        sp1.append(row['first_orders_period'])
        sp2.append(row['CRR'])

df=pd.DataFrame({'coh': sp1, 'CRR': sp2})
df.loc[df['CRR'].idxmax()]

# Ответ 5: когорта покупателей, совершивших покупку в июне 2017 года, является когортой с самым большим CRR на третий месяц: 0.45

# 6. Используя python, построить RFM-сегментацию пользователей, чтобы качественно оценить аудиторию. В кластеризации 
# выбрать следующие метрики: R - время от последней покупки пользователя до текущей даты, F - суммарное количество покупок
# у пользователя за всё время, M - сумма покупок за всё время. Подробно описать как созданы кластеры. Для каждого RFM-
# сегмента построить границы метрик recency, frequency и monetary для интерпретации этих кластеров.

merge_df_total_copy=merge_df_total.copy(deep=True)

# создание колонки с датой покупки в формате год-месяц 
merge_df_total_copy['delivered_date']=merge_df_total_copy['order_delivered_customer_date'].dt.strftime('%Y-%m') 
# фильтр по непустым датам доставки
delivered_df=merge_df_total_copy.query('order_delivered_customer_date != "NaT"')

# Чтобы определить период для rfm-анализа - посмотрим на кол-во пользователей и заказов по месяцам
by_month_2=delivered_df.groupby('delivered_date', as_index=False).agg({'customer_unique_id':'count', 'order_id':'nunique'})
plt.figure(figsize=(15,6))
plt.xticks(np.arange(len(by_month_2)), by_month_2['delivered_date'], rotation=90)
sns.lineplot(x='delivered_date', y ='customer_unique_id', data=by_month_2)
sns.lineplot(x='delivered_date', y ='order_id', data=by_month_2)

# В наших данных кол-во клиентов и заказов нарастает до сентября 2018 года. Для анализа RFM обычно используется год. Период с 
# сентября 2017 по август 2018 - наиболее интересен, т.к. на него приходится наибольшее кол-во пользователей и заказов.

# сформируем таблицу с заказами за нужный нам период
orders = delivered_df.query('delivered_date > "2017-08" and delivered_date < "2018-09"').groupby(['order_id', 'order_delivered_customer_date', 'customer_unique_id'], as_index=False).agg({'price': 'sum'})

# Для расчета R (времени с последней покупки) нам нужна текущая дата. Смоделируем проведение RFM-анализа на дату сразу после
# окончания периода
NOW = orders['order_delivered_customer_date'].max() + timedelta(days=1)

# Добавим столбец с количеством дней между покупкой и NOW. Далее чтобы найти значения R - нужно будет найти минимум этого столбца для каждого клиента.
orders['days_after_last_order'] = orders['order_delivered_customer_date'].apply(lambda x: (NOW - x).days)

# Сгруппируем данные по клиентам и посчитаем R, F и M
rfm = orders.groupby('customer_unique_id', as_index=False).agg({'days_after_last_order':'min','order_delivered_customer_date': 'count','price':'sum'})
rfm.rename(columns={'days_after_last_order': 'Recency', 'order_delivered_customer_date': 'Frequency', 'price':'Monetary'}, inplace=True)

# Чтобы вывести ранги для данных - необходимо разбить данные на диапазоны. Используем перцентили, отсекающие по 20%
# пользователей
level=[0.2, 0.4, 0.6, 0.8]
quantiles = rfm[['Recency', 'Frequency', 'Monetary']].quantile(level).to_dict()

# Присвоим ранги для Recency (меньшее значение недавности покупки - лучше)
intervals_R=[0]
for i in range(len(quantiles['Recency'])): 
    intervals_R.append(quantiles['Recency'][level[i]])
intervals_R.append(rfm.Recency.max())
rfm['R']=pd.cut(rfm.Recency, intervals_R, labels=['5', '4', '3', '2', '1'])

# % пользователей с одной покупкой
rfm.loc[rfm.Frequency==1].Frequency.count() / rfm.Frequency.count() *100

# Обратим внимание, что для Frequency метод перцентилей не подойдет (для всех перцентилей получаем одинаковые значения: 
# у нас 97% пользователей с одной покупкой). В данном случае не хватает информации и товаре, чтобы понимать как правильно
# ранжировать эту метрику. Возможно, у нас очень дорогой товар, который приобретается 1-2 раза, и клиенты с колличеством покупок
# более 5 - это премиум клиенты. А возможно, что сервис так плохо работает, что клиент, купив один раз, больше не возращается.
# Метрика Frequency находится в диапазоне от 1 до 11. Примем во внимание большой объем наблюдений и ранжируем следующим образом:
# [1-3] покупки ранг 1
# (3-5] покупок ранг 2
# (5-7] покупок ранг 3
# (7-9] покупок ранг 4
# (9-11] покупок ранг 5

# Присвоим ранги для Frequency 
intervals_F=[0, 3, 5, 7, 9, 12]
rfm['F']=pd.cut(rfm.Frequency, intervals_F, labels=['1', '2', '3', '4', '5'])

# Присвоим ранги для Monetary (чем больше сумма покупок, тем лучше)
intervals_M=[0]
for i in range(len(quantiles['Monetary'])): 
    intervals_M.append(quantiles['Monetary'][level[i]])
intervals_M.append(rfm.Monetary.max())
rfm['M']=pd.cut(rfm.Monetary, intervals_M, labels=['1','2','3','4','5'])

# скомбинируем значения R, F, M
rfm=rfm.astype({'R':'string', 'F':'string', 'M':'string'})
rfm['RFM_score'] = rfm['R'] + rfm['F'] + rfm['M']
rfm.head(3)

# Для того, чтобы понять в какой сегмент попадают пользователи по метрике Frequency - посмотрим еще раз на частоту встречаемости. 
rfm.F.value_counts()

# При трех метриках и 5-бальной ранговой системе вариантов комбинаций будет 125. С таким кол-вом комбинаций, конечно, работать
# не удобно. Разобъем пользователей по сегментам, в зависимости от RFM_score. 
# Создадим столбец с рангами, чтобы затем заменить его на названия сегментов
rfm['Segment'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)

# В словарь, с помощью регулярных выражений, занесем правила, по которым будем делить пользователей на сегменты. Как писалось
# выше, сложно оценить по кол-ву покупок к какому сегменту отнести покупателя, не зная что за товар продается. Поэтому в каждый
# сегмент попали покупатели с одной покупкой (т.к. они составляют основную массу покупателей), в некоторых сегментах это ранги
# 1-2, в некоторых 1-5. Сегментирование происходит скорее по метрикам Recency и Monetary. 

segment_map = {
    r'[1-2][1-2][1-2]': 'бездействующие', # покупали давно, мало заказов, маленькая выручка 
    r'[1-2][1-2][3-4]': 'в зоне риска', # покупали давно, мало заказов, средняя выручка 
    r'[1-2][1-5]5': 'не можем потерять', # покупали давно, не мало заказов, высокая выручка 
    r'3[1-2][1-2]': 'спящие', # покупали относительно недавно, мало заказов, маленькая выручка 
    r'3[1-2]3': 'требуют внимание', # покупали относительно недавно, мало заказов, средняя выручка 
    r'[3-4][1-5][4-5]': 'лояльные пользователи', # покупали относительно недавно, не мало заказов, высокая выручка 
    r'4[1-5]1': 'многообещающие', # покупали недавно, не мало заказов, маленькая выручка 
    r'5[1-5]1': 'новые пользователи', # покупали недавно, не мало заказов, маленькая выручка 
    r'[4-5][1-5][2-3]': 'потенциально-лояльные', # покупали недавно, не мало заказов, средняя выручка 
    r'5[1-5][4-5]': 'чемпионы'} # покупали недавно, не мало заказов, высокая выручка 

# заменим ранги на названия сегментов
rfm['Segment'] = rfm['Segment'].replace(segment_map, regex=True)

# посчитаем кол-во пользователей в получившихся сегментах и построим диаграмму
segmentation=rfm.groupby('Segment', as_index=False).agg({'customer_unique_id':'count'}).rename(columns={'customer_unique_id':'count_customer'}).sort_values('count_customer', ascending=False)
segmentation

fig = px.treemap(segmentation, path=['Segment'], values='count_customer')
fig.update_layout(title="Tree map of Segments", width=700, height=500,)
fig.show()
plt.figure(figsize=(15,6))
sns.barplot(x='count_customer', y='Segment', data = segmentation)

# Для вывода описания границ интервалов для метрик определим функцию
def description(score):
    r = int(score[0])
    f = int(score[1])
    m = int(score[2])
    rstr = "от " + str(intervals_R[r-1]) + " до " + str(intervals_R[r]) + " дней с даты последней покупки "
    fstr = "от " + str(intervals_F[f-1]) + " до " + str(intervals_F[f]) + " покупок в год "
    mstr = "от " + str(intervals_M[m-1]) + " до " + str(intervals_M[m]) + " руб. в год "
    return ("Границы метрик R: " + rstr + " F: "+ fstr+ " M: "+ mstr)

# создадим столбец с описанием
rfm['description_intervals'] = rfm['RFM_score'].apply(description) 

# Сброс ограничений на количество символов в записи
pd.set_option('display.max_colwidth', None)

# Описание границ метрик, по которым сотрудники по работе с покупателями смогут:
#     а) понять к какому сегменту относится пользователь
#     б) настроить работу с пользователями, в зависимости от сегмента
    
rfm.groupby(['RFM_score', 'Segment', 'description_intervals'], as_index=False).agg({'customer_unique_id':'count'}).rename(columns={'customer_unique_id': 'count_customers'})



