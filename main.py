import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.compose import make_column_selector


def filter_data(df):
    """
    Удаляет строки из датафрейма
    :param df:
    :return:
    """
    columns_for_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long']
    return df.drop(columns_for_drop, axis=1)


def year_prepare(df):
    """
    Убирает выбросы из колонки год
    :param df:
    :return:
    """
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        calc_bounds = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

        return calc_bounds

    boundaries = calculate_outliers(df['year'])
    df.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])
    return df


def feature_eng(df):
    """
    Создаёт признаки на основе имеющихся данных
    :param df:
    :return:
    """
    def short_model(x):
        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x

    # Добавляем фичу "short_model" – это первое слово из колонки model
    df.loc[:, 'short_model'] = df['model'].apply(short_model)

    # Добавляем фичу "age_category" (категория возраста)
    df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))

    return df


def main():
    data = pd.read_csv('data/30.5 homework.csv')
    df = data.copy()
    X = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    data_preparing = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('year_del', FunctionTransformer(year_prepare)),
        ('short_model', FunctionTransformer(feature_eng)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transform, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transform, make_column_selector(dtype_include=object))
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(bootstrap=False, class_weight= 'balanced', random_state=12),
        MLPClassifier(activation='logistic',max_iter = 500,  hidden_layer_sizes=(256, 128, 64))
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preparing', data_preparing),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, 'loan_pipe.pkl')




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
