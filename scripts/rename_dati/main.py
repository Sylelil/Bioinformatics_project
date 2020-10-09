
import os
from pathlib import Path
from shutil import copy2
import json

def get_dict(data):
    _dict = {}
    for item in data:
        _dict[item['file_name']] = item['cases'][0]['case_id']

    return _dict

def main():
    copyfiles('sani')
    copyfiles('malati')

def copyfiles(type):
    main_dir = Path('../..') / 'bioing' / type / ('dati_' + type)
    dest_dir = Path('../..') / 'bioing' / 'dati_genomica'
    with open(main_dir/'..'/'dati1.json') as json_file:
        data = json.load(json_file)
        _dict = get_dict(data)
        print(_dict)

    if type == 'sani':
        type_number = '0'
    else:
        type_number = '1'

    for dir in os.scandir(main_dir):
        if dir.is_dir():
            for gen_file in os.scandir(dir):
                if gen_file.name.endswith('txt.gz'):
                    file_name = _dict[gen_file.name]+'_'+type_number+'.txt.gz'
                    copy2(gen_file, dest_dir/file_name)


#if __name__ == '__main__':
    #main()


