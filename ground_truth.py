import torch
import clip
import os

from transformers import AutoProcessor, ClapModel, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from tqdm import tqdm
from sentence_transformers import util

# from atr_example import AudioDataset, load_audio_model
from audio import AudioDataset
from datatype import DataType
from nldbs import read_csv_furniture, read_csv_netflix_movies
from nlfilter import AudioProcessor, NLFilter, TextProcessor, ImageProcessor
from schema import NLColumn
from image import CraigslistDataset
import config_tdb


def craigslist_image():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50', device)

    img_dir = 'craigslist/furniture_imgs/'
    # img_dir = 'flickr/flickr30k_images/'
    img_paths = [img_dir + f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    t = transforms.Compose([
        transforms.ToPILImage()
    ])
    dataset = CraigslistDataset(img_paths, t)
    print('The nth image in the dataset: ', dataset[49][0])
    processor = ImageProcessor(dataset, model, preprocess, device)

    col = NLColumn('img', DataType.IMG, processor)

    # Find ground truth
    while True:
        text = input("Enter your query: ")
        nl_filter = NLFilter(col, text)
        nl_filter.update(1)
        while len(nl_filter.unsure_idxs()) > 0:
            nl_filter.collect_user_feedback(0.5)
            print(f'{nl_filter.text} {nl_filter._lower} {nl_filter._upper}')
        print(f'{nl_filter.text} {nl_filter.lower} {nl_filter.upper} {len(nl_filter.true_idxs())} {len(nl_filter.false_idxs())} {col.processor.nr_total}')


def craigslist_text(text=None, ground_truth=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print(device, device_id)
    
    df = read_csv_furniture()

    # Text.
    if config_tdb.USE_BART:
        text_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id)
    else:
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
    title_processor = TextProcessor(df['title'], text_model, device)
    col = NLColumn('title_u', DataType.NUM, title_processor)

    # Find ground truth
    is_text_given = text is not None
    while True:
        if not is_text_given:
            text = input("Enter your query: ") # wood
        nl_filter = NLFilter(col, text)
        start = time.time()
        nl_filter.update(1)
        end = time.time()
        print(f'Finished processing: {end - start}')
        nr_feedbacks = 0
        while len(nl_filter.unsure_idxs()) > 0:
            nr_feedbacks += 1
            nl_filter.collect_user_feedback(0.5, ground_truth)
            print(f'{nr_feedbacks} {nl_filter.text} {nl_filter._lower} {nl_filter._upper}')
        print(f'Final: {nr_feedbacks} {nl_filter.text} {nl_filter.lower} {nl_filter.upper} {len(nl_filter.true_idxs())} {len(nl_filter.false_idxs())} {col.processor.nr_total}')
        if is_text_given:
            break


def youtubeaudios_text(text=None, ground_truth=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print(device, device_id)
    # Read youtube table.
    df = pd.read_csv('audiocaps/youtube.csv', na_filter=False)
    df['audio'] = np.arange(len(df))

    # Initialize objects.
    # Text.
    if config_tdb.USE_BART:
        text_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id)
    else:
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
    description_processor = TextProcessor(df['description'], text_model, device)
    col = NLColumn('description', DataType.TEXT, description_processor)

    # Find ground truth
    is_text_given = text is not None
    while True:
        if not is_text_given:
            text = input("Enter your query: ")  # cooking
        nl_filter = NLFilter(col, text)
        # Process all data.
        start = time.time()
        nl_filter.update(1)
        end = time.time()
        print(f'Finished processing: {end - start}')
        nr_feedbacks = 0
        while nl_filter.nr_unsure() > 0:
            nr_feedbacks += 1
            nl_filter.collect_user_feedback(0.5, ground_truth)
            print(f'{nr_feedbacks} {nl_filter.text} {nl_filter._lower} {nl_filter._upper}')
        print(f'Final: {nr_feedbacks} {nl_filter.text} {nl_filter._lower} {nl_filter._upper} {len(nl_filter.true_idxs())} {len(nl_filter.false_idxs())} {col.processor.nr_total}')
        if is_text_given:
            break


def netflix_text(text=None, ground_truth=None, col_name='featured_review'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print(device, device_id)
    
    df = read_csv_netflix_movies()

    # Text.
    if config_tdb.USE_BART:
        text_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id)
    else:
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_processor = TextProcessor(df[col_name], text_model, device)
    col = NLColumn(col_name + '_u', DataType.NUM, text_processor)

    # Find ground truth
    is_text_given = text is not None
    while True:
        if not is_text_given:
            text = input("Enter your query: ") # wood
        nl_filter = NLFilter(col, text)
        start = time.time()
        nl_filter.update(1)
        end = time.time()
        print(f'Finished processing: {end - start}')
        nr_feedbacks = 0
        while len(nl_filter.unsure_idxs()) > 0:
            nr_feedbacks += 1
            nl_filter.collect_user_feedback(0.5, ground_truth)
            print(f'{nr_feedbacks} {nl_filter.text} {nl_filter._lower} {nl_filter._upper}')
        print(f'Final: {nr_feedbacks} {nl_filter.text} {nl_filter.lower} {nl_filter.upper} {len(nl_filter.true_idxs())} {len(nl_filter.false_idxs())} {col.processor.nr_total}')
        if is_text_given:
            break

def youtubeaudios_audio():
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    # Read youtube table.
    df = pd.read_csv('audiocaps/youtube.csv', na_filter=False)
    # Load audio dataset and processor for the audio column. Only include valid audios.
    dataset = AudioDataset(valid_idxs=df['audio'])

    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    if device_id >= 0:
        model = model.to(device_id)
    preprocess = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    processor = AudioProcessor(dataset, model, preprocess, device_id)

    df['audio'] = np.arange(len(df))

    # Initialize objects.
    # Audio.
    col = NLColumn('audio', DataType.AUDIO, processor)

    # Find ground truth
    while True:
        text = input("Enter your query: ")  # voices
        nl_filter = NLFilter(col, text)
        nr_unsure = 0
        while nr_unsure > 0 or len(nl_filter.idx_to_score) < nl_filter.col.processor.nr_total:
            if nr_unsure > 0:
                nl_filter.collect_user_feedback(0.5)
                print(f'{nl_filter.text} {nl_filter.lower} {nl_filter.upper} {len(nl_filter.true_idxs())} {len(nl_filter.false_idxs())} {len(nl_filter.idx_to_score)} {col.processor.nr_total}')
            else:
                # Total of 46774 audio files. Processing 0.1: 8346.204621315002s
                start = time.time()
                nl_filter.update(0.01)
                end = time.time()
                print(f'Processing: {end - start}')
            nr_unsure = nl_filter.nr_unsure()
        print(f'Final: {nl_filter.text} {nl_filter.lower} {nl_filter.upper} {len(nl_filter.true_idxs())} {len(nl_filter.false_idxs())} {col.processor.nr_total}')


def measure_youtubeaudios():
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    # Read youtube table.
    df = pd.read_csv('audiocaps/youtube.csv', na_filter=False)
    # Load audio dataset and processor for the audio column. Only include valid audios.
    dataset = AudioDataset(valid_idxs=df['audio'])

    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    if device_id >= 0:
        model = model.to(device_id)
    preprocess = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    
    df['audio'] = np.arange(len(df))
    # Batch loader.
    data_loader = DataLoader(dataset=dataset, batch_size=64)

    text = 'voices'
    start = time.time()
    scores_list = []
    for batch_id, (audios, *_) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # audios = audios.to(device, torch.float32)
        # audio_embeds, caption_embeds = model(audios, text)
        inputs = preprocess(text=text, audios=audios, sampling_rate=48000, return_tensors="pt", padding=True)
        if device_id >= 0:
            inputs = inputs.to(device_id)
        outputs = model(**inputs)
        scores = 100. * util.cos_sim(outputs.audio_embeds, outputs.text_embeds)
        scores_list.append(scores)
    end = time.time()
    print(f'Finished processing: {end - start}')


if __name__ == "__main__":
    # craigslist_image()
    # craigslist_text()
    # craigslist_text(text='wood', ground_truth=0.9013)  # BART
    # youtubeaudios_text()
    # youtubeaudios_text(text='cooking', ground_truth=0.8506) # BART
    # youtubeaudios_audio()
    # measure_youtubeaudios()
    netflix_text()
