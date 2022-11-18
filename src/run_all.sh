#!/bin/bash

# Q0
python main.py --model bert-med --train_percentages 100 --val_percentages 100 --train_dataset amazon_books --val_dataset amazon_books --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 100 --val_percentages 80,20 --train_dataset amazon_books --val_dataset amazon_books,amazon_video --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 100 --val_percentages 60,40 --train_dataset amazon_books --val_dataset amazon_books,amazon_video --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 100 --val_percentages 40,60 --train_dataset amazon_books --val_dataset amazon_books,amazon_video --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 100 --val_percentages 20,80 --train_dataset amazon_books --val_dataset amazon_books,amazon_video --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 100 --val_percentages 100 --train_dataset amazon_books --val_dataset amazon_video --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000

python main.py --model bert-med --train_percentages 80,20 --val_percentages 80,20 --train_dataset amazon_books --val_dataset amazon_books,amazon_video --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 60,40 --val_percentages 60,40 --train_dataset amazon_books,amazon_video --val_dataset amazon_books,amazon_video --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 40,60 --val_percentages 40,60 --train_dataset amazon_books,amazon_video --val_dataset amazon_books,amazon_video --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 20,80 --val_percentages 20,80 --train_dataset amazon_books,amazon_video --val_dataset amazon_books,amazon_video --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 100 --val_percentages 100 --train_dataset amazon_video --val_dataset amazon_books --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000

python main.py --model bert-med --train_percentages 80,20 --val_percentages 100 --train_dataset amazon_books,amazon_video --val_dataset amazon_books --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 60,40 --val_percentages 100 --train_dataset amazon_books,amazon_video --val_dataset amazon_books --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 40,60 --val_percentages 100 --train_dataset amazon_books,amazon_video --val_dataset amazon_books --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 20,80 --val_percentages 100 --train_dataset amazon_books,amazon_video --val_dataset amazon_books --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
python main.py --model bert-med --train_percentages 100 --val_percentages 100 --train_dataset amazon_video --val_dataset amazon_books --device cuda --mode all,last,first,middle --n_train 5000 --n_val 1000
# Q1
# python3 icl.py --task icl  --model med,full --dataset babi --k 0,1,16
# python3 icl.py --task icl  --model med,full --dataset xsum --k 0,1,4 --prompt none,tldr,custom

# # Q2
# python3 ft.py --task ft   --model med --mode first,last,middle,lora4,lora16 --dataset xsum,babi --k 0,1,8,128

# # Q3
# python3 icl.py --task icl  --model med --dataset babi --k 16 --repeats 5
# python3 ft.py --task ft   --model med --dataset babi --k 16 --repeats 5 --mode lora16
