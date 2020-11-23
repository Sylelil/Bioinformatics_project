import os
from pathlib import Path
import json
from shutil import copy2

def get_dict(data):
    _inv_dict = {}
    for item in data:
        _inv_dict[item['cases'][0]['case_id']] = item['file_name']

    return _inv_dict

def main():
    copyfiles('sani')
    copyfiles('malati')

def copyfiles(type):
    main_dir = Path('/Users/elisa/PycharmProjects/script/sani/immagini')
    with open(main_dir/'..'/'immagini.json') as json_file:
        data = json.load(json_file)
        _dict = get_dict(data)
        print(_dict)

    _new_dict = {value:key for(key,value) in _dict.items()}
    print(_new_dict)

    if type == 'sani':
        type_number = '0'
    else:
        type_number = '1'

    for dir in os.scandir(main_dir):
        if dir.is_dir():
            #print(dir)
            for img_file in os.scandir(dir):
                #print(img_file)
                if img_file.name.endswith('.svs'):
                    if img_file.name in _new_dict:
                        file_name =  _new_dict[img_file.name]+'_'+type_number+'.svs'
                        #print(file_name)
                        print(f"join: {img_file} {os.path.join(dir, file_name)}")
                        os.rename(img_file, os.path.join(dir, file_name))
                    else:
                        print(f"renaming {img_file} in _deleted_{img_file.name}")
                        #os.rename(img_file, os.path.join(dir, "_deleted_" + img_file.name))
                    break



if __name__ == '__main__':
    main()
