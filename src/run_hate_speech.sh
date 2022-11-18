python main.py --model bert-med --train_percentages 100 --val_percentages 100 --train_dataset civil_comments --val_dataset civil_comments --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 100 --val_percentages 80,20 --train_dataset civil_comments --val_dataset civil_comments,tweet_eval --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 100 --val_percentages 60,40 --train_dataset civil_comments --val_dataset civil_comments,tweet_eval --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 100 --val_percentages 40,60 --train_dataset civil_comments --val_dataset civil_comments,tweet_eval --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 100 --val_percentages 20,80 --train_dataset civil_comments --val_dataset civil_comments,tweet_eval --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 100 --val_percentages 100 --train_dataset civil_comments --val_dataset tweet_eval --device cuda --mode all,last,first,middle

python main.py --model bert-med --train_percentages 80,20 --val_percentages 80,20 --train_dataset civil_comments --val_dataset civil_comments,tweet_eval --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 60,40 --val_percentages 60,40 --train_dataset civil_comments,tweet_eval --val_dataset civil_comments,tweet_eval --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 40,60 --val_percentages 40,60 --train_dataset civil_comments,tweet_eval --val_dataset civil_comments,tweet_eval --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 20,80 --val_percentages 20,80 --train_dataset civil_comments,tweet_eval --val_dataset civil_comments,tweet_eval --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 100 --val_percentages 100 --train_dataset tweet_eval --val_dataset civil_comments --device cuda --mode all,last,first,middle

python main.py --model bert-med --train_percentages 80,20 --val_percentages 100 --train_dataset civil_comments,tweet_eval --val_dataset civil_comments --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 60,40 --val_percentages 100 --train_dataset civil_comments,tweet_eval --val_dataset civil_comments --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 40,60 --val_percentages 100 --train_dataset civil_comments,tweet_eval --val_dataset civil_comments --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 20,80 --val_percentages 100 --train_dataset civil_comments,tweet_eval --val_dataset civil_comments --device cuda --mode all,last,first,middle
python main.py --model bert-med --train_percentages 100 --val_percentages 100 --train_dataset tweet_eval --val_dataset civil_comments --device cuda --mode all,last,first,middle