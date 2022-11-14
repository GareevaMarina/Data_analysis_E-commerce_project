# E-commerce-project
Sales analysis, rfm-segmentation of customers, cohort analysis

## Дана некоторая платформа онлайн продаж, с которой выгружены следующие данные:
 * уникальные идентификаторы пользователей
 * заказы
 * товарные позиции, входящие в заказы
 
## Необходимо проанализировать данные и ответить на следующие вопросы:
1. Сколько пользователей, которые совершили покупку только один раз?
2. Сколько заказов в месяц в среднем не доставляется по разным причинам (вывести детализацию по причинам)? 
3. По каждому товару определить, в какой день недели товар чаще всего покупается. 
4. Сколько у каждого из пользователей в среднем покупок в неделю (по месяцам)?
5. Провести когортный анализ пользователей. В период с января по декабрь выявить когорту с самым высоким retention на 3й месяц
6. Построить RFM-сегментацию пользователей, чтобы качественно оценить аудиторию.

### Анализ проведен на Python, выводы в файле E-commerce_project.ipynb

## Тепловая карта метрики retention в когортном анализе
![Screenshot_29](https://user-images.githubusercontent.com/104904113/201678228-a177b44c-8cc4-4edc-8c0f-f25cc000bf23.jpg)

## Дерево сегментов, на которые разбиты пользователи при проведении RFM-сегментации
![Screenshot_30](https://user-images.githubusercontent.com/104904113/201678619-4a5b3839-ba38-476f-8ff7-48ada6f31189.jpg)

## Количество пользователей по сегментам при проведении RFM-сегментации
![Screenshot_31](https://user-images.githubusercontent.com/104904113/201678796-39ff4e7e-dc04-40a8-a58d-fc1102318d0c.jpg)
