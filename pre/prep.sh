#!bin/bash
for file in $(ls data/images)
do 
    echo "file is $file"
    python preprocess.py $file data/images_new/

done 