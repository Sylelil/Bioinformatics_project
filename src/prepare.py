import gzip
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from config import paths
import json


def get_dict(data):
    _dict = {}
    for item in data:
        _dict[item['file_name']] = item['cases'][0]['case_id']
    return _dict


def prepare_files_class(data_class, data_type, dir, json_dir, dest_dir):
    main_dir = dir
    data_dir = Path(dir) / data_class
    json_f = Path(json_dir) / f'{data_class}.json'

    with open(json_f) as json_file:  # open(main_dir/'..'/'dati1.json') as json_file:
        data = json.load(json_file)
        _dict = get_dict(data)

    if data_type == 'images':
        extension_name = '.svs'
    else:
        extension_name = '.txt.gz'

    if data_class == 'normal':
        type_number = '0'
    else:
        type_number = '1'

    for d in tqdm(os.listdir(data_dir), desc=">> Reading data...", file=sys.stdout):
        current_dir = os.path.join(data_dir, d)
        if os.path.isdir(current_dir):
            for file in os.listdir(current_dir):
                file_path = os.path.join(current_dir, file)
                if file.endswith(extension_name):
                    # if genes, unzip:
                    if data_type == 'genes':
                        new_filename = _dict[file] + '_' + type_number + extension_name
                        new_filename = new_filename.replace(".gz", "")
                        with gzip.open(file_path, 'rb') as f_in:
                            with open(Path(dest_dir)/ new_filename, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        os.remove(file_path)
                    else:
                        new_filename = _dict[file] + '_' + type_number + extension_name
                        os.rename(file_path, Path(dest_dir)/ new_filename)


def prepare_files(data_type, dir, json_dir, dest_dir):
    prepare_files_class('normal', data_type, dir, json_dir, dest_dir)
    prepare_files_class('tumor', data_type, dir, json_dir, dest_dir)


def get_array(json_file):
    # with open('E:\\bioing\\malati\\dati1.json') as dati1:
    with open(json_file) as dati1:
        data = json.load(dati1)
        result = map(lambda item: item['cases'][0]['case_id'], data)
        return list(result)


def filter_files_class(data_class, dest_images_dir, dest_genes_dir, images_json_path, genes_json_path):
    genes = get_array(genes_json_path)
    images = get_array(images_json_path)
    intersezione = list(set(genes) & set(images))

    filenames_genes = []
    for file in tqdm(os.listdir(dest_genes_dir), desc=">> Reading data...", file=sys.stdout):
        cid = file.replace(f'_{data_class}.txt', "")
        if file.endswith(f'_{data_class}.txt'):
            if cid not in intersezione:
                os.remove(Path(dest_genes_dir) / file)
            else:
                filenames_genes.append(file)

    duplicates = [k for k, v in Counter(filenames_genes).items() if v > 1]
    if len(duplicates) >= 1:
        print(f"error: found {len(duplicates)} duplicated caseids: {duplicates}")
    for dup in duplicates:
        i = 0
        for file in tqdm(os.listdir(dest_genes_dir), desc=">> Reading data...", file=sys.stdout):
            if file == dup:
                if i == 0:
                    i = 1
                    continue
                else:
                    os.remove(Path(dest_genes_dir) / f'{dup}_{data_class}.txt')

    filenames_images = []
    for file in tqdm(os.listdir(dest_images_dir), desc=">> Reading data...", file=sys.stdout):
        cid = file.replace(f'_{data_class}.svs', "")
        if file.endswith(f'_{data_class}.svs'):
            if cid not in intersezione:
                os.remove(Path(dest_images_dir) / file)
            else:
                filenames_images.append(cid)

    duplicates = [k for k, v in Counter(filenames_images).items() if v > 1]
    if len(duplicates) >= 1:
        print(f"error: found {len(duplicates)} duplicated caseids: {duplicates}")
    for dup in duplicates:
        i = 0
        for file in tqdm(os.listdir(dest_images_dir), desc=">> Reading data...", file=sys.stdout):
            if file == dup:
                if i == 0:
                    i = 1
                    continue
                else:
                    os.remove(Path(dest_images_dir) / f'{dup}_{data_class}.svs')


def filter_files(dest_images_dir, images_json_dir, dest_genes_dir, genes_json_dir):
    filter_files_class('0', dest_images_dir, dest_genes_dir, Path(images_json_dir) / 'normal.json', Path(genes_json_dir) / 'normal.json')
    filter_files_class('1', dest_images_dir, dest_genes_dir, Path(images_json_dir) / 'tumor.json', Path(genes_json_dir) / 'tumor.json')


def main():
    original_images_dir = paths.original_images_dir
    original_genes_dir = paths.original_genes_dir
    images_json_dir = paths.images_json_dir
    genes_json_dir = paths.genes_json_dir
    dest_images_dir = paths.images_dir
    dest_genes_dir = paths.genes_dir
    if not os.path.exists(dest_genes_dir):
        os.makedirs(dest_genes_dir)
    if not os.path.exists(dest_images_dir):
        os.makedirs(dest_images_dir)

    prepare_files('images', original_images_dir, images_json_dir, dest_images_dir)
    prepare_files('genes', original_genes_dir, genes_json_dir, dest_genes_dir)

    filter_files(dest_images_dir, images_json_dir, dest_genes_dir, genes_json_dir)


if __name__ == '__main__':
    main()


