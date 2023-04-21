import pandas as pd
import numpy as np
from datetime import date, datetime

import src.util as util


def camel(snake):
    """
    Function to change a string to lower camel case:
    This function is based on
    stackoverflow.com/questions/19053707/converting-snake-case-to-lower-camel-case-lowercamelcase
    """
    if isinstance(snake, str):
        first, *others = snake.split('_')
        return ''.join([first.lower(), *map(str.title, others)])
    else:
        return snake


def date2dayType(date_s):
    date_dt = datetime.strptime(date_s, '%Y-%m-%d')

    weekno = date_dt.weekday()

    if weekno < 5:
        return "weekday"
    else:  # 5 Sat, 6 Sun
        return "weekend"


def date2weekday(date_s):
    date_dt = datetime.strptime(date_s, '%Y-%m-%d')

    weekday = date_dt.weekday()

    return ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][weekday]


def date2season(date_s):
    """
    convert a date to season, based on
    https://stackoverflow.com/questions/16139306/determine-season-given-timestamp-in-python-using-datetime
    """
    Y = 2004  # dummy leap year to allow input X-02-29 (leap day)
    seasons = [('winter', (date(Y, 1, 1), date(Y, 3, 20))),
               ('spring', (date(Y, 3, 21), date(Y, 6, 20))),
               ('summer', (date(Y, 6, 21), date(Y, 9, 22))),
               ('autumn', (date(Y, 9, 23), date(Y, 12, 20))),
               ('winter', (date(Y, 12, 21), date(Y, 12, 31)))]

    date_dt = datetime.strptime(date_s, '%Y-%m-%d').date().replace(year=Y)
    # print( date_dt)
    # print( next(season for season, (start, end) in seasons
    #            if (start <= date_dt) and (date_dt <= end)) )

    return next(season for season, (start, end) in seasons
                if (start <= date_dt) and (date_dt <= end))


def filter_min_count(df, column, threshold):
    original = df[column].nunique()
    itemCounts = df.value_counts(column)
    remainingItems = set(itemCounts.index[itemCounts >= threshold])
    df = df[df[column].isin(remainingItems)]
    remaining = df[column].nunique()
    return df, original - remaining


def k_core(df, user_id, item_id, min_items_per_user, min_users_per_item):
    while True:
        df, users_removed = filter_min_count(df, user_id, min_items_per_user)
        df, items_removed = filter_min_count(df, item_id, min_users_per_item)
        if users_removed == 0 and items_removed == 0:
            break

    return df
