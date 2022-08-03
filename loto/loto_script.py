import logging
import requests
import zipfile
import os
import pandas as pd
import numpy as np
from itertools import chain
from collections import Counter
import random
import argparse

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

URLS = [
    'https://media.fdj.fr/static/csv/loto/loto_197605.zip',
    'https://media.fdj.fr/static/csv/loto/loto_200810.zip',
    'https://media.fdj.fr/static/csv/loto/loto_201703.zip',
    'https://media.fdj.fr/static/csv/loto/loto_201902.zip',
    'https://media.fdj.fr/static/csv/loto/loto_201911.zip',
]

CSVS = [f'loto_{i}.csv' for i in range(5)]

COLUMNS = ['annee_numero_de_tirage'] + [f'boule_{i}' for i in range(1, 6)] + ['numero_chance']

def asint(l):
    return [int(e) for e in l]

def custom_fn(row):
    return (sorted(asint(row[1:6])), [int(row[6])])

def read_csv(filename):
    data = pd.read_csv(filename, sep=';')
    if filename == CSVS[0]:
        columns = ['annee_numero_de_tirage'] + [f'boule_{i}' for i in range(1, 6)] + ['boule_complementaire']
        data = data[columns]
        data.columns = COLUMNS
        data.numero_chance = data.numero_chance.map(lambda x: x % 11)

    else:
        data = data[COLUMNS]

    return data

def generate_games(games, history, n_games):

    if history > len(games):
        logger.warning(f'History value exceeds the number of history games, setting history to maximal value: {len(games)}')
        history = len(games)


    logger.info(f'Generating {n_games} games based on the last past {history} games:')

    generated_games = []

    past_balls = chain(*[game[0] for game in games[:history]])
    past_stars = chain(*[game[1] for game in games[:history]])

    ball_counter = Counter(past_balls)
    star_counter = Counter(past_stars)

    played_balls = np.array(list(ball_counter.keys()))
    played_stars = np.array(list(star_counter.keys()))

    proba_balls = np.array(list(ball_counter.values())) / np.linalg.norm(list(ball_counter.values()), ord=1)
    proba_stars = np.array(list(star_counter.values())) / np.linalg.norm(list(star_counter.values()), ord=1)


    for i in range(n_games):

        while True:
            ball_selection = sorted(np.random.choice(played_balls, size=5, replace=False, p=proba_balls))
            star_selection = sorted(np.random.choice(played_stars, size=1, replace=False, p=proba_stars))

            if sum([ball > 25 for ball in ball_selection]) != random.choice([2,3]):
                continue
            else:
                generated_games.append((ball_selection, star_selection))
                break

    return generated_games

def download_helper(download_all = False):
    for i, url in enumerate(URLS):
        zipfilename = f'loto_{i}.zip'
        csvfilename = f'loto_{i}.csv'
        if not(os.path.isfile(csvfilename)) or download_all:
            logger.info(f'Downloading file {csvfilename}')
            r = requests.get(url, allow_redirects=True)
            open(zipfilename, 'wb').write(r.content)
            zipdata = zipfile.ZipFile(zipfilename)
            zipinfo = zipdata.infolist()[0]
            zipinfo.filename = csvfilename
            zipdata.extract(zipinfo)
            if os.path.isfile(zipfilename):
                os.remove(zipfilename)


def download_data(download_all=False):
    if download_all:
        download_helper(download_all)
    else:
        if not all([os.path.isfile(csv) for csv in CSVS]):
            logger.info('Some files are missing, we are downloading the files')
            download_helper(download_all)

if __name__=='__main__':


    # parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--ngames', nargs='?', type=int, default=2, help='Number of games to generate')
    argparser.add_argument('--history', nargs='?', type=int, default=50, help='Number of previous games to consider')
    argparser.add_argument('--download', dest='download', action='store_true', help='Boolean to decide wether you should or not download all the files')
    argparser.set_defaults(download=False)

    args = argparser.parse_args()
    download_data(args.download)

    total_games = pd.concat([read_csv(csv) for csv in CSVS], axis=0).sort_values(by='annee_numero_de_tirage', ascending=False)
    games = total_games.apply(custom_fn, axis = 1).values.tolist() # format loto data

    gen_games = generate_games(games, args.history, args.ngames)

    for game in gen_games:
        print(game)