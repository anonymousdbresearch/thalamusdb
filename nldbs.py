import time
import torch
import clip
import os

from transformers import AutoProcessor, ClapModel, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from torchvision import transforms
import duckdb

import config_tdb
from audio import AudioDataset
# from atr_example import AudioDataset, load_audio_model
from image import CraigslistDataset
from nlfilter import ImageProcessor, TextProcessor, AudioProcessor
from schema import NLDatabase, NLTable, NLColumn, DataType

dfs = {}
models = {}


def get_df_by_name(table_name):
    if table_name in dfs:
        return dfs[table_name]
    else:
        if table_name == 'furniture':
            df = read_csv_furniture()
        elif table_name == 'youtube':
            df = read_csv_youtube()
        elif table_name == 'movies':
            df = read_csv_netflix_movies()
        elif table_name == 'ratings':
            df = read_csv_netflix_ratings()
        else:
            raise ValueError(f'Wrong table name: {table_name}')
        dfs[table_name] = df
        return df


def get_text_model(device_id):
    if 'text' not in models:
        if config_tdb.USE_BART:
            models['text'] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id)
        else:
            models['text'] = SentenceTransformer('all-MiniLM-L6-v2')
    return models['text']


def get_image_model(device):
    if 'image' not in models:
        models['image'] = clip.load('RN50', device)
    return models['image']


# def get_audio_model_atr(device):
#     if 'audio' not in models:
#         models['audio'] = load_audio_model(device)
#     return models['audio']


def get_audio_model(device_id):
    if 'audio' not in models:
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        if device_id >= 0:
            model = model.to(device_id)
        preprocess = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        models['audio'] = (model, preprocess)
    return models['audio']


def get_nldb_by_name(dbname):
    if dbname == 'craigslist':
        return craigslist()
    elif dbname == 'youtubeaudios':
        return youtubeaudios()
    elif dbname == 'netflix':
        return netflix()
    else:
        raise ValueError(f'Wrong nldb name: {dbname}')


def read_csv_furniture():
    df = pd.read_csv('craigslist/furniture.tsv', sep='\t', index_col=0)
    df['aid'] = np.arange(len(df))
    df['title_u'] = np.arange(len(df))
    df.drop('imgs', axis=1, inplace=True)
    df['time'] = pd.to_datetime(df['time']).astype(np.int64) / 10 ** 9
    return df


def read_csv_youtube():
    df = pd.read_csv('audiocaps/youtube.csv')
    df['description'] = df['description'].fillna('')
    df['likes'] = df['likes'].fillna(0)
    df['description_u'] = np.arange(len(df))
    df['audio'] = np.arange(len(df))
    return df


def read_csv_netflix_movies():
    # df = pd.read_csv('netflix/movies.csv')
    # df.columns = [col.lower() for col in df.columns]
    # df['movietitle_u'] = np.arange(len(df))
    df = pd.read_csv('netflix/movies_with_reviews.csv')
    df = df.drop('review_label', axis=1, errors='ignore')
    df['featured_review_u'] = np.arange(len(df))
    return df


def read_csv_netflix_ratings():
    df = pd.read_csv('netflix/ratings.csv')
    df.columns = [col.lower() for col in df.columns]
    return df


def get_craiglist_images():
    # Load image dataset and processor for img column.
    img_dir = 'craigslist/furniture_imgs/'
    # img_dir = 'flickr/flickr30k_images/'
    img_paths = [img_dir + f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    t = transforms.Compose([
        transforms.ToPILImage()
    ])
    return CraigslistDataset(img_paths, t)


def add_count_columns(relationships, name2df):
    for table_dim, col_dim, table_fact, col_fact in relationships:
        df_fact = name2df[table_fact]
        df_counts = df_fact[col_fact].value_counts().reset_index()
        col_count = col_dim + '_c'
        df_counts.columns = [col_dim, col_count]

        df_dim = name2df[table_dim]
        merged_df = df_dim.merge(df_counts, on=col_dim, how='left')
        merged_df[col_count] = merged_df[col_count].fillna(0)
        name2df[table_dim] = merged_df
    return name2df


def craigslist():
    print(f'Initializing NL Database: Craigslist')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    model, preprocess = get_image_model(device)

    dataset = get_craiglist_images()
    processor = ImageProcessor(dataset, model, preprocess, device)

    # Read furniture table from csv.
    df_furniture = get_df_by_name('furniture')
    nr_imgs = len(dataset)
    print(f'len(images): {nr_imgs}')
    df_images = pd.DataFrame([[idx, int(dataset[idx][1].split('_')[0])] for idx in range(nr_imgs)], columns=['img', 'aid'])

    # For foreign key relationships, add an extra column.
    name2df = {'furniture': df_furniture, 'images': df_images}
    relationships = [('furniture', 'aid', 'images', 'aid')]
    name2df = add_count_columns(relationships, name2df)

    # For web interface.
    # # Initialize tables in duckdb.
    # db_path = 'db/craigslist.db'
    # is_first_init = not os.path.isfile(db_path)
    # con = duckdb.connect(db_path) #, check_same_thread=False)
    # # Register furniture table.
    # if is_first_init:
    #     con.execute("CREATE TABLE furniture AS SELECT * FROM df")
    #     con.execute("CREATE UNIQUE INDEX furniture_aid_idx ON furniture (aid)")
    #     con.commit()
    # print(f'len(furniture): {len(df)}')
    # # Register image table.
    # if is_first_init:
    #     con.execute("CREATE TABLE images(img INTEGER PRIMARY KEY, aid INTEGER)")
    #     con.executemany("INSERT INTO images VALUES (?, ?)",
    #                 [[idx, dataset[idx][1].split('_')[0]] for idx in range(nr_imgs)])
    #     con.commit()
    # nr_imgs = len(dataset)
    # print(f'len(images): {nr_imgs}')
    
    # Initialize tables in duckdb.
    con = duckdb.connect(database=':memory:') #, check_same_thread=False)
    for name, df in name2df.items():
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
    con.execute("CREATE UNIQUE INDEX furniture_aid_idx ON furniture (aid)")
    print(f'len(furniture): {len(df_furniture)}')

    # Load text dataset and processor for the title column.
    text_model = get_text_model(device_id)
    title_processor = TextProcessor(df_furniture['title'], text_model, device)

    # Create NL database.
    furniture = NLTable('furniture')
    furniture.add(NLColumn('aid', DataType.NUM),
                  NLColumn('time', DataType.NUM),
                  NLColumn('neighborhood', DataType.TEXT),
                  NLColumn('title', DataType.TEXT),
                  NLColumn('title_u', DataType.NUM, title_processor),
                  NLColumn('url', DataType.TEXT),
                  NLColumn('price', DataType.NUM))
    images = NLTable('images')
    images.add(NLColumn('img', DataType.IMG, processor),
               NLColumn('aid', DataType.NUM))
    nldb = NLDatabase('craigslist', con)
    nldb.add(furniture, images)
    nldb.add_relationships(*relationships)
    for table_dim, col_dim, _, _ in relationships:
        nldb.tables[table_dim].add(NLColumn(f'{col_dim}_c', DataType.NUM))
    # Initialize metadata information.
    nldb.init_info()
    return nldb


def youtubeaudios():
    print(f'Initializing NL Database: YoutubeAudios')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    # Read youtube table.
    df = get_df_by_name('youtube')
    # Load audio dataset and processor for the audio column. Only include valid audios.
    dataset = AudioDataset(valid_idxs=df['audio'])
    model, preprocess = get_audio_model(device_id)
    processor = AudioProcessor(dataset, model, preprocess, device_id)
    # model = get_audio_model(device)
    print(f'GPU - Audio Model: {next(model.parameters()).is_cuda}')
    # processor = AudioProcessor(dataset, model, device)

    # For web interface.
    # db_path = 'db/youtube.db'
    # is_first_init = not os.path.isfile(db_path)
    # con = duckdb.connect(db_path)  #, check_same_thread=False)
    # if is_first_init:
    #     con.execute("CREATE TABLE youtube AS SELECT * FROM df")
    #     con.commit()

    # Register youtube table. Should be done after updating the audio column.
    con = duckdb.connect(database=':memory:') #, check_same_thread=False)
    con.execute("CREATE TABLE youtube AS SELECT * FROM df")

    # Load text dataset and processor for the description column.
    text_model = get_text_model(device_id)
    description_processor = TextProcessor(df['description'], text_model, device)

    # Create NL database.
    youtube = NLTable('youtube')
    youtube.add(NLColumn('youtube_id', DataType.TEXT),
                NLColumn('audio', DataType.AUDIO, processor),
                NLColumn('title', DataType.TEXT),
                NLColumn('category', DataType.TEXT),
                NLColumn('viewcount', DataType.NUM),
                NLColumn('author', DataType.TEXT),
                NLColumn('length', DataType.NUM),
                NLColumn('duration', DataType.TEXT),
                NLColumn('likes', DataType.NUM),
                NLColumn('description', DataType.TEXT),
                NLColumn('description_u', DataType.NUM, description_processor))
    nldb = NLDatabase('youtubeaudios', con)
    nldb.add(youtube)
    # Initialize metadata information.
    nldb.init_info()
    return nldb


def netflix():
    print(f'Initializing NL Database: Netflix')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    start = time.time()
    # Read movie table.
    df_movie = get_df_by_name('movies')
    # Read rating table.
    df_rating = get_df_by_name('ratings')
    end = time.time()
    print(f'Finished reading csv: {end - start}')

    # For foreign key relationships, add an extra column.
    name2df = {'movies': df_movie, 'ratings': df_rating}
    relationships = [('movies', 'movieid', 'ratings', 'movieid')]
    start = time.time()
    name2df = add_count_columns(relationships, name2df)
    end = time.time()
    print(f'Finished adding column for foreign key relationship: {end - start}')

    # Register tables.
    start = time.time()
    con = duckdb.connect(database=':memory:') #, check_same_thread=False)
    for name, df in name2df.items():
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
    end = time.time()
    print(f'Finished registering tables: {end - start}')

    # Load text dataset and processor for the description column.
    text_model = get_text_model(device_id)
    # print(f'GPU - Text Model: {next(text_model.parameters()).is_cuda}')
    title_processor = TextProcessor(df_movie['movietitle'], text_model, device)
    featured_review_processor = TextProcessor(df_movie['featured_review'], text_model, device)

    # Create NL database.
    movies = NLTable('movies')
    movies.add(NLColumn('movieid', DataType.NUM),
               NLColumn('releaseyear', DataType.NUM),
               NLColumn('movietitle', DataType.TEXT),
               NLColumn('movietitle_u', DataType.NUM, title_processor),
               NLColumn('featured_review', DataType.TEXT),
               NLColumn('featured_review_u', DataType.NUM, featured_review_processor))
    ratings = NLTable('ratings')
    ratings.add(NLColumn('custid', DataType.NUM),
                NLColumn('rating', DataType.NUM),
                NLColumn('date', DataType.TEXT),
                NLColumn('movieid', DataType.NUM))
    nldb = NLDatabase('netflix', con)
    nldb.add(movies, ratings)
    nldb.add_relationships(*relationships)
    for table_dim, col_dim, _, _ in relationships:
        nldb.tables[table_dim].add(NLColumn(f'{col_dim}_c', DataType.NUM))
    # Initialize metadata information.
    nldb.init_info()
    return nldb



