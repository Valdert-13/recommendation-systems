import pandas as pd
import numpy as np


def prefilter_items(data, item_features, purchases_weeks=22, take_n_popular=5000):
    """Пред-фильтрация товаров
    Input
    -----
    data: pd.DataFrame
        Датафрейм с информацией о покупках
    item_features: pd.DataFrame
        Датафрейм с информацией о товарах
    """

    # Уберем товары с нулевыми продажами
    data = data[data['quantity'] != 0]

    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 5 месяцев
    purchases_last_week = data.groupby('item_id')['week_no'].max().reset_index()
    weeks = purchases_last_week[
        purchases_last_week['week_no'] > data['week_no'].max() - purchases_weeks].item_id.tolist()
    data = data[data['item_id'].isin(weeks)]

    # Уберем не интересные для рекоммендаций категории (department)
    department_size = pd.DataFrame(item_features.groupby('department')['item_id'].nunique(). \
                                   sort_values(ascending=False)).reset_index()

    department_size.columns = ['department', 'n_items']

    rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
    items_in_rare_departments = item_features[
        item_features['department'].isin(rare_departments)].item_id.unique().tolist()

    data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] >= 0.7]

    # Уберем слишком дорогие товары
    data = data[data['price'] < 50]

    # Возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер не покупил товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data


