import json
import os
import shutil
from collections import Counter


def get_array_genomics():
    with open('E:\\bioing\\malati\\dati1.json') as dati1:
        data = json.load(dati1)
        result = map(lambda item: item['cases'][0]['case_id'], data)
        return list(result)


def get_array_images():
    with open('E:\\bioing\\malati\\immagini.json') as immagini:
        data = json.load(immagini)
        result = map(lambda item: item['cases'][0]['case_id'], data)
        return list(result)


def main():
    dati = get_array_genomics()
    immagini = get_array_images()
    intersezione = list(set(dati) & set(immagini))

    _dir_dati = r'E:\\bioing\\dati_genomica'
    _dir_immagini = 'E:\\bioing\\malati\\immagini_renamed'

    cids= []
    for gen_file in os.scandir(_dir_dati):
        cid = gen_file.path.replace('_1.txt.gz',"")
        cids.append(cid)
        if gen_file.name.endswith('_1.txt.gz'):
            if gen_file.name.replace('_1.txt.gz',"") not in intersezione:
                os.remove(gen_file)

    duplicates = [k for k, v in Counter(cids).items() if v > 1]
    if len(duplicates) >= 1:
        print(f"error: found {len(duplicates)} duplicated caseids: {duplicates}")
    for dup in duplicates:
        os.remove(dup+'_1.txt.gz')

    cids_im = []
    for imm_dir in os.scandir(_dir_immagini):
        for imm_file in os.scandir(imm_dir):
            cid = imm_file.path.replace('_1.txt.gz',"") # todo perché .txt.gz se sono immagini?
            cids_im.append(cid)
            if imm_file.name.endswith('.svs') and imm_file.name.replace('_1.svs',"") not in intersezione:
                shutil.rmtree(imm_dir)

    duplicates = [k for k, v in Counter(cids_im).items() if v > 1]
    if len(duplicates) >= 1:
        print(f"error: found {len(duplicates)} duplicated caseids: {duplicates}")
    for dup in duplicates:
        os.remove(dup+'_1.txt.gz')


if __name__ == '__main__':
    main()
