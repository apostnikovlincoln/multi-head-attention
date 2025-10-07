# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 14:02:02 2025

@author: Andrey Postnikov
"""

from torch.utils.data import Dataset
import pandas as pd
import os
import glob

# Pickling CSV data
def csv_to_pickled_chunks(input_csv: str, output_dir: str, chunk_size: int, max_chunks: int = None):
    """
    Reads the presence-abscence matrix in chunks and pickles each chunk as a pandas DataFrame.

    Parameters
    ----------
    input_csv : str
        Path to the CSV file.
    output_dir : str
        Directory where the pickled chunks will be saved.
    chunk_size : int
        Number of rows per chunk.
    max_chunks : int, optional
        Maximum number of chunks to process (default: None = process all).

    Returns
    -------
    None
    """
    # Check if output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV in chunks
    reader = pd.read_csv(input_csv,  delimiter=",", on_bad_lines='warn', chunksize=chunk_size)

    for i, chunk in enumerate(reader, start=1):
        if max_chunks is not None and i > max_chunks:
            print(f"Stopped after {max_chunks} chunks.")
            break
        output_path = os.path.join(output_dir, f"matrix_chunk_{i}.pkl")
        chunk.to_pickle(output_path)
        print(f"Saved chunk {i} with shape {chunk.shape} to {output_path}.")


# Unpickling serialised data
def load_chunks(folder: str, start: int, end: int, concat: bool = True):
    """
    Load a range of pickled pandas DataFrame chunks.

    Parameters
    ----------
    folder : str
        Path to the directory containing the .pkl files.
    start : int
        First chunk index to load (inclusive).
    end : int
        Last chunk index to load (inclusive).
    concat : bool, default=True
        If True, concatenate all chunks into a single DataFrame.
        If False, return a list of DataFrames.

    Returns
    -------
    DataFrame or list of DataFrames
    """
    dfs = []
    for i in range(start, end + 1):
        path = os.path.join(folder, f"matrix_chunk_{i}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chunk {i} not found at {path}")
        print(f"Loading {path} ...")
        dfs.append(pd.read_pickle(path))
    
    if concat:
        return pd.concat(dfs, ignore_index=True)
    else:
        return dfs


def get_chunk_metadata(folder: str):
    """
    Summarise metadata (row counts, columns, total rows) of pickled chunks.
    """

    chunk_files = sorted(glob.glob(os.path.join(folder, "matrix_chunk_*.pkl")))
    metadata = []
    for f in chunk_files:
        df = pd.read_pickle(f)
        metadata.append({"file": f, "rows": len(df), "columns": list(df.columns)})
    total_rows = sum(m["rows"] for m in metadata)
    return {"chunks": metadata, "total_rows": total_rows}


def load_filtered_chunks(folder: str, filter_func, start: int = None, end: int = None):
    """
    Load chunks and filter rows using a custom function without loading all rows into memory.

    Parameters
    ----------
    filter_func : callable
        Function that takes a DataFrame and returns a filtered DataFrame.
    """
    dfs = []
    chunk_files = sorted(os.listdir(folder))
    for idx, f in enumerate(chunk_files, start=1):
        if start and idx < start:
            continue
        if end and idx > end:
            break
        df = pd.read_pickle(os.path.join(folder, f))
        dfs.append(filter_func(df))
    return pd.concat(dfs, ignore_index=True)


def filter_func_example(df, idx=[0]):
    # TODO
    return

# Apply a tokenizer chunk by chunk instead of loading all data at once.
def tokenize_chunks(folder: str, tokenizer, text_column: str, output_dir: str, max_length: int = 512):
    """
    Tokenize text data in chunks for transformer input.

    Parameters
    ----------
    folder : str
        Path to pickled chunks.
    tokenizer : transformers.PreTrainedTokenizer
        Hugging Face tokenizer.
    text_column : str
        Column containing text to tokenize.
    output_dir : str
        Where to save tokenized numpy arrays or pickled token dictionaries.
    max_length : int
        Maximum sequence length for padding/truncation.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_files = sorted(os.listdir(folder))

    for i, f in enumerate(chunk_files, start=1):
        df = pd.read_pickle(os.path.join(folder, f))
        encodings = tokenizer(
            df[text_column].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np"
        )
        out = os.path.join(output_dir, f"tokenized_{i}.pkl")
        pd.to_pickle(encodings, out)
        print(f"Saved tokenized chunk {i} → {out}")
        
# Wrap partial datasets in a PyTorch Dataset class      
class ChunkedDataset(Dataset):
    def __init__(self, folder: str, start: int = None, end: int = None):
        self.folder = folder
        self.chunk_files = sorted(os.listdir(folder))
        if start:
            self.chunk_files = self.chunk_files[start-1:]
        if end:
            self.chunk_files = self.chunk_files[:end]

        self.data = []
        for f in self.chunk_files:
            chunk = pd.read_pickle(os.path.join(folder, f))
            self.data.extend(chunk if isinstance(chunk, list) else [chunk])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Set paths to input and output directories, create an output directory if necessary
input_dir = "../Data/E_coli_pangenome_files/"
output_dir = "../Data/E_coli_pangenome_files/pickled_chunks"
os.makedirs(output_dir, exist_ok=True)

# Number of rows to serialise per file
chunk_size = 10000

# Serialise data from the presense-absence matrix
csv_to_pickled_chunks(input_dir+'gene_presence_absence_modified.csv', output_dir, chunk_size)

# Load serialised data (e.g. chunks from 1 to 3)
chunks = load_chunks(output_dir, 1, 3, False)