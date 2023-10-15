# ThalamusDB: Approximate Query Processing on Multi-Modal Data

> ThalamusDB is an approximate query processing system that processes complex SQL queries on multi-modal data. Our data model extends the relational model and integrates multi-modal data, including visual, audio, and text data, as columns. Users can write SQL queries including predicates on multi-modal data, described in natural language.

## Quick Start

```bash
# Tested on Python 3.10.
# Git LFS required to download large files like *.zip.
git clone https://github.com/anonymousdbresearch/thalamusdb.git
cd thalamusdb

# Unzipped image files: 186MB
unzip craigslist/furniture_imgs.zip -d craigslist

# Unzipped audio files: 32GB
# To run the YouTube benchmark, download the raw waveforms of the AudioCaps dataset: https://audiocaps.github.io/
# Then, unzip files into train, test, and val folders in audiocaps/waveforms

# Unzipped csv file: 2.7GB
unzip netflix/ratings.zip -d netflix

# (Optional) Create virtual environment.
python -m venv .venv
source .venv/bin/activate

# Install requirements.
pip install -r requirements.txt
# Install CLIP (on a CUDA GPU machine). CUDA Toolkit required.
pip install git+https://github.com/openai/CLIP.git
# Run our benchmark.
python benchmark.py
```

Note that, to run the YouTube benchmark with the audio model used in our experiments, you would need to download and setup the audio model: https://github.com/akoepke/audio-retrieval-benchmark. The current setting of this repository uses a different audio-text model, [CLAP](https://arxiv.org/abs/2211.06687), as the underlying audio model.

## How to Integrate New Datasets

There are two ways to integrate new datasets.

### Using Command-Line Tool.

Run `console.py` to create a new database. Use `CREATE TABLE` statements to create tables with columns of `IMAGE`, `AUDIO`, `TEXT`, and `INTEGER` data types. Add foreign key constraints by using the `ALTER TABLE` command. Use `COPY` statements to insert rows from a csv file. Run queries with `SELECT` statements with natural language predicates on multi-modal data using the `NL` keyword.

Example:
```sql
CREATE TABLE furniture(time INTEGER, neighborhood TEXT, title TEXT, url TEXT, price INTEGER, aid INTEGER);
CREATE TABLE images(img IMAGE, aid INTEGER);
ALTER TABLE images ADD FOREIGN KEY (aid) REFERENCES furniture (aid);
COPY furniture FROM 'craigslist/formated_furniture.csv' DELIMITER ',';
COPY images FROM 'craigslist/formated_imgs.csv' DELIMITER ',';
SELECT max(price) FROM images, furniture WHERE images.aid = furniture.aid AND nl(img, 'wooden');
```

### Using an NLDatabase instance.

Create a new function in `nldbs.py` that creates a NLDatabase instance (refer to functions `craisglist()` and `youtubeaudios()`). It requires loading relational data to DuckDB and providing pointers to image and audio data.