import json
import os
import shutil

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

    for gen_file in os.scandir(_dir_dati):
        if gen_file.name.endswith('_1.txt.gz'):
            if gen_file.name.replace('_1.txt.gz',"") not in intersezione:
                os.remove(gen_file)

    for imm_dir in os.scandir(_dir_immagini):
        for imm_file in os.scandir(imm_dir):
            if imm_file.name.endswith('.svs') and imm_file.name.replace('_1.svs',"") not in intersezione:
                shutil.rmtree(imm_dir)

if __name__ == '__main__':
    main()
