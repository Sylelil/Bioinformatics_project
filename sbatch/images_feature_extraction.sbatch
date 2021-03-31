#!/bin/bash
#SBATCH --job-name=tiles_to_numpy
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s279901@studenti.polito.it
#SBATCH --partition=global
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=tiles_to_numpy.log
#SBATCH --mem-per-cpu=16384M

cp -r ../src $TMPDIR/src
cp -r ../datasets $TMPDIR/datasets
cp -r ../requirements.txt $TMPDIR

module load intel/python/3.5
virtualenv bioinf3
source bioinf3/bin/activate

apt-get install -y openslide-tools && apt-get install -y python-openslide
pip install -r requirements.txt

mkdir $TMPDIR/extracted_features && mkdir $TMPDIR/extracted_features/images && mkdir $TMPDIR/extracted_features/images/numpy_normal && mkdir $TMPDIR/extracted_features/images/numpy_tumor

python3 -u $TMPDIR/src/images_feature_extraction.py

cp -r $TMPDIR/extracted_features /home/bioinfo_group_03/Bioinformatics_project/