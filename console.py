import os
import re

import clip
import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
from torchvision import transforms

import config_tdb
from datatype import DataType
from nldbs import get_craiglist_images
from query_info import NLQuery
from schema import NLColumn, NLDatabase, NLTable
from image import ImageDataset
from nlfilter import ImageProcessor, TextProcessor, AudioProcessor


class Console:
    def __init__(self):
        self.con = duckdb.connect(database=':memory:')
        self.table2cols = {}
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
        self.nldb = NLDatabase('temp_db', self.con)

    
    def get_text_model(self):
        if 'text' not in self.models:
            if config_tdb.USE_BART:
                self.models['text'] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=self.device_id)
            else:
                self.models['text'] = SentenceTransformer('all-MiniLM-L6-v2')
        return self.models['text']


    def get_image_model(self):
        if 'image' not in self.models:
            model, preprocess = clip.load('RN50', self.device)
            t = transforms.Compose([transforms.ToPILImage()])
            self.models['image'] = (model, preprocess, t) 
        return self.models['image']


    # def get_audio_model(self):
    #     if 'audio' not in self.models:
    #         self.models['audio'] = load_audio_model(self.device)
    #     return self.models['audio']

    def run(self):
        query = ""
        while True:
            line = input("Enter SQL query (or '\q' to quit): ")
            if line.lower() == '\q':
                break
            query += line + " "

            if query.strip().endswith(';'):
                query = ' '.join(query.split()).strip()
                if query.strip().upper().startswith('CREATE TABLE '):
                    self.create_table(query)
                elif query.strip().upper().startswith('COPY '):
                    self.copy_csv(query)
                elif query.strip().upper().startswith('ALTER TABLE '):
                    self.alter_table(query)
                elif query.strip().upper().startswith('SELECT '):
                    nl_query = NLQuery(query)
                    constraint = ('error', 0.1, 1)
                    self.nldb.run(nl_query, constraint, optimizer_mode='local')
                else:
                    print(f"Invalid query: {query}.")
                query = ""
    
    def alter_table(self, query):
        # Use regular expressions to parse ALTER TABLE ADD FOREIGN KEY statement
        alter_table_pattern = r'ALTER\s+TABLE\s+(\w+)\s+ADD\s+FOREIGN\s+KEY\s+\(([^)]+)\)\s+REFERENCES\s+(\w+)\s+\(([^)]+)\);'
        match = re.match(alter_table_pattern, query.strip(), re.IGNORECASE)
        if not match:
            print("Invalid ALTER TABLE ADD FOREIGN KEY statement.")
            return
        
        table_name = match.group(1)
        foreign_key_columns = match.group(2).split(',')[0]
        referenced_table_name = match.group(3)
        referenced_columns = match.group(4).split(',')[0]

        print(f"Table Name: {table_name}")
        print(f"Foreign Key Columns: {foreign_key_columns}")
        print(f"Referenced Table: {referenced_table_name}")
        print(f"Referenced Columns: {referenced_columns}")

        self.nldb.add_relationships((referenced_table_name, referenced_columns, table_name, foreign_key_columns))

    def create_table(self, query):
        # Use regular expressions to parse CREATE TABLE statement
        create_table_pattern = r'CREATE\s+TABLE\s+(\w+)\s*\(([^)]+)\);'
        match = re.match(create_table_pattern, query.strip(), re.IGNORECASE)
        if not match:
            print("Invalid CREATE TABLE statement.")
            return
        
        table_name = match.group(1)
        columns = match.group(2).split(',')
        column_infos = []
        for column in columns:
            parts = column.strip().split()
            column_name = parts[0]
            column_type = ' '.join(parts[1:])
            column_infos.append((column_name, column_type))
        if not column_infos:
            print("No columns found in the CREATE TABLE statement.")
            return
        self.table2cols[table_name] = column_infos

        print(f"Table Name: {table_name}")
        print("Columns:")
        for name, data_type in column_infos:
            print(f"- Name: {name}, Type: {data_type}")
        
        # # Create table in duckdb.
        # converted_column_infos = []
        # for name, data_type in column_infos:
        #     if data_type.upper() in ['IMAGE', 'AUDIO', 'TEXT']:
        #         converted_column_infos.append((name, 'INTEGER'))
        #     else:
        #         converted_column_infos.append((name, data_type))

        # self.con.execute(f"CREATE TABLE {table_name} ({','.join([f'{name} {data_type}' for name, data_type in converted_column_infos])})")
    
    def copy_csv(self, query):
        # Use regular expressions to parse COPY statement
        copy_pattern = r"COPY\s+(\w+)\s+FROM\s+'([^']+)'\s+DELIMITER\s+'([^']+)';"
        match = re.match(copy_pattern, query.strip(), re.IGNORECASE)
        if not match:
            print("Invalid COPY statement.")
            return

        table_name = match.group(1)
        file_path = match.group(2)
        delimiter = match.group(3)
        print(f"Table Name: {table_name}")
        print(f"File Path: {file_path}")
        print(f"Delimiter: {delimiter}")

        # # Copy csv file into duckdb.
        df = pd.read_csv(file_path, sep=delimiter)

        table = NLTable(table_name)
        for name, data_type in self.table2cols[table_name]:
            file_paths = df[name]
            if data_type.upper() == 'IMAGE':
                model, preprocess, t = self.get_image_model()
                dataset = ImageDataset(file_paths, t)
                img_processor = ImageProcessor(dataset, model, preprocess, self.device)
                table.add(NLColumn(name, DataType.IMG, img_processor))
                df[name] = np.arange(len(df))
            elif data_type.upper() == 'AUDIO':
                # dataset = AudioDataset(valid_idxs=df[name])
                # model = self.get_audio_model()
                # audio_processor = AudioProcessor(dataset, model, self.device)
                # table.add(NLColumn(name, DataType.AUDIO, audio_processor))
                # df[name] = np.arange(len(df))
                pass
            elif data_type.upper() == 'TEXT':
                text_model = self.get_text_model()
                text_processor = TextProcessor(df[name], text_model, self.device)
                table.add(NLColumn(name, DataType.TEXT))
                table.add(NLColumn(name + "_u", DataType.NUM, text_processor))
                df[name + "_u"] = np.arange(len(df))
            else:
                table.add(NLColumn(name, DataType.NUM))
        self.nldb.add(table)
        # Register table.
        self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        print(f'len(furniture): {len(df)}')
        # Initialize metadata information.
        self.nldb.init_info()


def format_craigslist_tables():
    # Table funiture.
    df = pd.read_csv('craigslist/furniture.tsv', sep='\t', index_col=0)
    df['aid'] = np.arange(len(df))
    df.drop('imgs', axis=1, inplace=True)
    df['time'] = pd.to_datetime(df['time']).astype(np.int64) / 10 ** 9
    df.to_csv('craigslist/formated_furniture.csv', index=False)

    # Table images.
    # Load image dataset and processor for img column.
    img_dir = 'craigslist/furniture_imgs/'
    img_paths = [img_dir + f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    dataset = get_craiglist_images()
    image_rows = [[img_paths[idx], dataset[idx][1].split('_')[0]] for idx in range(len(dataset))]
    pd.DataFrame(image_rows, columns=['img', 'aid']).to_csv('craigslist/formated_imgs.csv', index=False)


# CREATE TABLE furniture(time INTEGER, neighborhood TEXT, title TEXT, url TEXT, price INTEGER, aid INTEGER);
# CREATE TABLE images(img IMAGE, aid INTEGER);
# ALTER TABLE images ADD FOREIGN KEY (aid) REFERENCES furniture (aid);
# COPY furniture FROM 'craigslist/formated_furniture.csv' DELIMITER ',';
# COPY images FROM 'craigslist/formated_imgs.csv' DELIMITER ',';
# select max(price) from images, furniture where images.aid = furniture.aid and nl(img, 'wooden');
if __name__ == "__main__":
    Console().run()
    # format_craigslist_tables()
