import numpy as np
import pandas as pd


def train_test_preprocessing(data, train, recommender, N):
    """Подготовка обучающего датасета, разбиение на X и y"""

    users_lvl_2 = pd.DataFrame(data['user_id'].unique())

    users_lvl_2.columns = ['user_id']

    train_users = train['user_id'].unique()
    users_lvl_2 = users_lvl_2[users_lvl_2['user_id'].isin(train_users)]

    # Рекомендации на основе собственных покупок
    users_lvl_2_ = users_lvl_2.copy()
    users_lvl_2['candidates'] = users_lvl_2['user_id'].apply(
        lambda x: recommender.get_own_recommendations(x, N=N)
    )

    s = users_lvl_2.apply(
        lambda x: pd.Series(x['candidates']), axis=1
    ).stack().reset_index(level=1, drop=True)

    s.name = 'item_id'

    users_lvl_2 = users_lvl_2.drop('candidates', axis=1).join(s)

    users_lvl_2['flag'] = 1

    targets_lvl_2 = data[['user_id', 'item_id']].copy()
    targets_lvl_2.head(2)

    targets_lvl_2['target'] = 1  # тут только покупки

    targets_lvl_2 = users_lvl_2.merge(targets_lvl_2, on=['user_id', 'item_id'], how='left')

    targets_lvl_2['target'].fillna(0, inplace=True)
    targets_lvl_2.drop('flag', axis=1, inplace=True)

    return targets_lvl_2


def new_item_features(data, item_features, items_emb_df):
    """Новые признаки для продуктов"""
    new_item_features = item_features.merge(data, on='item_id', how='left')

    ##### Добавим имбеддинги
    item_features = item_features.merge(items_emb_df, how='left')

    ##### discount
    mean_disc = new_item_features.groupby('item_id')['coupon_disc'].mean().reset_index().sort_values('coupon_disc')
    item_features = item_features.merge(mean_disc, on='item_id', how='left')

    ###### manufacturer
    rare_manufacturer = item_features.manufacturer.value_counts()[item_features.manufacturer.value_counts() < 50].index
    item_features.loc[item_features.manufacturer.isin(rare_manufacturer), 'manufacturer'] = 999999999
    item_features.manufacturer = item_features.manufacturer.astype('object')

    ##### 1 Количество продаж и среднее количество продаж товара
    item_qnt = new_item_features.groupby(['item_id'])['quantity'].count().reset_index()
    item_qnt.rename(columns={'quantity': 'quantity_of_sales'}, inplace=True)

    item_qnt['quantity_of_sales_per_week'] = item_qnt['quantity_of_sales'] / new_item_features['week_no'].nunique()
    item_features = item_features.merge(item_qnt, on='item_id', how='left')

    ##### 2 Среднее количество продаж товара в категории в неделю
    items_in_department = new_item_features.groupby('department')['item_id'].count().reset_index().sort_values(
        'item_id', ascending=False
    )
    items_in_department.rename(columns={'item_id': 'items_in_department'}, inplace=True)

    qnt_of_sales_per_dep = new_item_features.groupby(['department'])['quantity'].count().reset_index().sort_values(
        'quantity', ascending=False
    )
    qnt_of_sales_per_dep.rename(columns={'quantity': 'qnt_of_sales_per_dep'}, inplace=True)

    items_in_department = items_in_department.merge(qnt_of_sales_per_dep, on='department')
    items_in_department['qnt_of_sales_per_item_per_dep_per_week'] = (
            items_in_department['qnt_of_sales_per_dep'] /
            items_in_department['items_in_department'] /
            new_item_features['week_no'].nunique()
    )
    items_in_department = items_in_department.drop(['items_in_department'], axis=1)
    item_features = item_features.merge(items_in_department, on=['department'], how='left')

    ##### sub_commodity_desc
    items_in_department = new_item_features.groupby('sub_commodity_desc')['item_id'].count().reset_index().sort_values(
        'item_id', ascending=False
    )
    items_in_department.rename(columns={'item_id': 'items_in_sub_commodity_desc'}, inplace=True)

    qnt_of_sales_per_dep = new_item_features.groupby(['sub_commodity_desc'])[
        'quantity'].count().reset_index().sort_values(
        'quantity', ascending=False
    )
    qnt_of_sales_per_dep.rename(columns={'quantity': 'qnt_of_sales_per_sub_commodity_desc'}, inplace=True)

    items_in_department = items_in_department.merge(qnt_of_sales_per_dep, on='sub_commodity_desc')
    items_in_department['qnt_of_sales_per_item_per_sub_commodity_desc_per_week'] = (
            items_in_department['qnt_of_sales_per_sub_commodity_desc'] /
            items_in_department['items_in_sub_commodity_desc'] /
            new_item_features['week_no'].nunique()
    )
    items_in_department = items_in_department.drop(['items_in_sub_commodity_desc'], axis=1)
    item_features = item_features.merge(items_in_department, on=['sub_commodity_desc'], how='left')

    return item_features


def new_user_features(data, user_features, users_emb_df):
    """Новые признаки для пользователей"""

    new_user_features = user_features.merge(data, on='user_id', how='left')

    ##### Добавим имбеддинги
    user_features = user_features.merge(users_emb_df, how='left')


    ##### Обычное время покупки
    time = new_user_features.groupby('user_id')['trans_time'].mean().reset_index()
    time.rename(columns={'trans_time': 'mean_time'}, inplace=True)
    time = time.astype(np.float32)
    user_features = user_features.merge(time, how='left')


    ##### Возраст
    user_features['age'] = user_features['age_desc'].replace(
        {'65+': 70, '45-54': 50, '25-34': 30, '35-44': 40, '19-24':20, '55-64':60}
    )
    user_features = user_features.drop('age_desc', axis=1)


    ##### Доход
    user_features['income'] = user_features['income_desc'].replace(
        {'35-49K': 45,
     '50-74K': 70,
     '25-34K': 30,
     '75-99K': 95,
     'Under 15K': 15,
     '100-124K': 120,
     '15-24K': 20,
     '125-149K': 145,
     '150-174K': 170,
     '250K+': 250,
     '175-199K': 195,
     '200-249K': 245}
    )
    user_features = user_features.drop('income_desc', axis=1)


    ##### Дети
    user_features['kids'] = 0
    user_features.loc[(user_features['kid_category_desc'] == '1'), 'kids'] = 1
    user_features.loc[(user_features['kid_category_desc'] == '2'), 'kids'] = 2
    user_features.loc[(user_features['kid_category_desc'] == '3'), 'kids'] = 3
    user_features = user_features.drop('kid_category_desc', axis=1)


    ##### Средний чек, средний чек в неделю
    basket = new_user_features.groupby(['user_id'])['sales_value'].sum().reset_index()

    baskets_qnt = new_user_features.groupby('user_id')['basket_id'].count().reset_index()
    baskets_qnt.rename(columns={'basket_id': 'baskets_qnt'}, inplace=True)

    average_basket = basket.merge(baskets_qnt)

    average_basket['average_basket'] = average_basket.sales_value / average_basket.baskets_qnt
    average_basket['sum_per_week'] = average_basket.sales_value / new_user_features.week_no.nunique()

    average_basket = average_basket.drop(['sales_value', 'baskets_qnt'], axis=1)
    user_features = user_features.merge(average_basket, how='left')

    return user_features

def new_features(data, train, recommender, item_features, user_features, items_emb_df, users_emb_df, N=50):

    target = train_test_preprocessing(data, train, recommender, N)
    user_features = new_user_features(data, user_features, users_emb_df)
    item_features = new_item_features(data, item_features, items_emb_df)
    item_features = data.merge(item_features, on='item_id', how='left')

    ####################################################################################################################
    """Объеденяем датасет"""

    data_df = item_features.merge(user_features, on='user_id', how='left')

    """коэффициент количества покупок товаров в данной категории к среднему количеству"""
    df_1 = data_df.groupby(['user_id', 'commodity_desc', 'week_no']).agg({'quantity': 'mean'}) \
        .reset_index().rename(columns={'quantity': 'count_purchases_week_dep'})

    df_2 = data_df.groupby(['commodity_desc', 'week_no']).agg({'quantity': 'sum'}) \
        .reset_index().rename(columns=({'quantity': 'mean_count_purchases_week_dep'}))

    df = df_1.merge(df_2, on=['commodity_desc', 'week_no'], how='left')
    df['count_purchases_week_mean'] = df['count_purchases_week_dep'] / df['mean_count_purchases_week_dep']
    df = df[['user_id', 'commodity_desc', 'count_purchases_week_mean']]

    temp_df = df.groupby(['user_id', 'commodity_desc']).agg({'count_purchases_week_mean': 'mean'}) \
        .reset_index()

    data_df = data_df.merge(temp_df, on=['user_id', 'commodity_desc'], how='left')

    """коэффициент отношения суммы покупок товаров в данной категории к средней сумме"""
    df_1 = data_df.groupby(['user_id', 'commodity_desc', 'week_no']).agg({'sales_value': 'sum'}) \
        .reset_index().rename(columns={'sales_value': 'sales_value_week'})

    df_2 = data_df.groupby(['commodity_desc', 'week_no']).agg({'sales_value': 'sum'}) \
        .reset_index().rename(columns=({'sales_value': 'mean_sales_value_week'}))

    df = df_1.merge(df_2, on=['commodity_desc', 'week_no'], how='left')
    df['sum_purchases_week_mean'] = df['sales_value_week'] / df['mean_sales_value_week']
    df = df[['user_id', 'commodity_desc', 'sum_purchases_week_mean']]

    temp_df = df.groupby(['user_id', 'commodity_desc']).agg({'sum_purchases_week_mean': 'mean'}) \
        .reset_index()

    data_df = data_df.merge(temp_df, on=['user_id', 'commodity_desc'], how='left')

    data_df = data_df.merge(target, on=['item_id', 'user_id'], how='left')
    data_df = data_df.fillna(0)
    # data_df = data_df.loc[data_df['target'].notna()]

    return data_df





def get_important_features(model, X_train, y_train):
    """Список важных признаков"""
    model.fit(X_train, y_train)
    feature_imp = list(zip(X_train.columns.tolist(), model.feature_importances_))
    feature_imp = pd.DataFrame(feature_imp, columns=['feature', 'value'])
    basic_feats = feature_imp.loc[feature_imp.value > 0, 'feature'].tolist()
    return basic_feats


def popularity_recommendation(data, n=5):
    """Топ-n популярных товаров"""

    popular = data.groupby('item_id')['quantity'].count().reset_index()
    popular.sort_values('quantity', ascending=False, inplace=True)
    popular = popular[popular['item_id'] != 999999]
    recs = popular.head(n).item_id
    return recs.tolist()

def get_different_categories(list_recommendations, item_info):
    """Получение списка товаров из уникальных категорий"""
    finish_recommendations = []

    categories_used = []

    CATEGORY_NAME = 'sub_commodity_desc'

    for item in list_recommendations:
        category = item_info.loc[item_info['item_id'] == item, CATEGORY_NAME].values[0]

        if category not in categories_used:
            finish_recommendations.append(item)
            categories_used.append(category)

    return finish_recommendations


def get_final_recomendation_list(row, item_info, train_1, N=5):
    """Пост-фильтрация товаров

    Input
    -----
    row: строка датасета
    item_info: pd.DataFrame
        Датафрейм с информацией о товарах
     train_1: pd.DataFrame
        обучающий датафрейм
    """
    recommend = row['recomendations']
    purchased_goods = train_1.loc[train_1['user_id'] == row['user_id']]['item_id'].unique()
    

    df_price = train_1.groupby('item_id')['sales_value'].mean().reset_index()

    list_pop_rec = popularity_recommendation(train_1, n=300)

    if recommend == 0:
        recommend = list_pop_rec
    # 1 Уникальность
    unique_recommendations = []
    [unique_recommendations.append(item) for item in recommend if item not in unique_recommendations]

    # 2 Товары дороже 1$
    price_recommendations = []
    [price_recommendations.append(item) for item in unique_recommendations if df_price \
        .loc[df_price['item_id'] == item]['sales_value'].values > 1]

    # 3 один товар > 7 $
    expensive_items = []
    [expensive_items.append(item) for item in price_recommendations if df_price. \
        loc[df_price['item_id'] == item]['sales_value'].values > 7]

    # 4 товара юзер никогда не покупал
    new_items = []
    [new_items.append(item) for item in price_recommendations if item not in purchased_goods]

    # Промежуточный итог
    suppurt_rec = []
    suppurt_rec.append(expensive_items[0] if len(expensive_items) > 0 else list_pop_rec[0])
    suppurt_rec += new_items
    suppurt_rec = get_different_categories(suppurt_rec, item_info=item_info)[0:3]
    suppurt_rec += price_recommendations
    final_recommendations = get_different_categories(suppurt_rec, item_info=item_info)

    n_rec = len(final_recommendations)
    if n_rec < N:
        final_recommendations.extend(list_pop_rec[:N - n_rec])
    else:
        final_recommendations = final_recommendations[:N]

    assert len(final_recommendations) == N, 'Количество рекомендаций != {}'.format(N)
    return final_recommendations

def get_final_recomendation(X_test, test_preds_proba, val_2, train_1, item_features):

    """Финальный список рекомендованных товаров"""
    X_test['predict_proba'] = test_preds_proba

    X_test.sort_values(['user_id', 'predict_proba'], ascending=False, inplace=True)
    recs = X_test.groupby('user_id')['item_id']
    recomendations = []
    for user, preds in recs:
        recomendations.append({'user_id': user, 'recomendations': preds.tolist()})

    recomendations = pd.DataFrame(recomendations)

    result_2 = val_2.groupby('user_id')['item_id'].unique().reset_index()
    result_2.columns = ['user_id', 'actual']

    result = result_2.merge(recomendations, how='left')
    result['recomendations'] = result['recomendations'].fillna(0)

    result['recomendations'] = result.progress_apply \
        (lambda x: get_final_recomendation_list(x, item_info=item_features, train_1=train_1, N=5), axis=1)

    return result