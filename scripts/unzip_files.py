import os
import shutil

def main():

    dir_gen = ""
    final_path = os.path.join(dir_gen, "final_genes")
    os.mkdir(final_path)

    for gen_dir in os.scandir(dir_gen):
        for gen_file in os.scandir(gen_dir):
            if gen_file.name.endswith('.txt.gz'):
                new_file = gen_file.name.replace(".gz","")
                with gzip.open(gen_file, 'rb') as f_in:
                    with open(os.path.join(final_path,new_file), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    main()