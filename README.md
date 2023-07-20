# NLPSurgicalFineTuning

[Final poster]("CS 330 Project Poster - Coefficient-based fine-tuning of language models on distribution shift.jpg")

##
Coding: since one part of the project used similar datasets and coding as HW3, we based our code on HW3 code starter. We of course changed major parts for efficiency and to adapt to our idea. That includes the way we handle data (with dataloaders), the training loop, the saving of the experiments (with tensorboard logger) and other things like the model we build.

## Running a training

### Command line arguments

`--model`: NLP model to use for the training (we used BERT-tiny, mini, small and med).</br>
`--train_dataset`: list datasets to use for the training. The training will be done on a linear combination of those datasets, as described in our report. </br>
`--val_dataset`: list datasets to use for the validation. The validation will be done on a linear combination of those datasets, as described in our report. </br>
`--train_percentages`: list of percentages (sum must equal 1) for the train datasets. each percentage corresponds to the dataset at the same position in the list. For instance if `train_dataset=[amazon-books, amazon-video]` and `train_percentages=[80, 20]`, then the dataset for training would be 0.8*amazon-books + 20*amazon-video (see our report for more details).</br>
`--val_percentages`: list of percentages (sum must equal 1) for the val datasets. each percentage corresponds to the dataset at the same position in the list. Works the same as `train_percentages`. </br>
`--mode`: mode with which to fine-tune (available: all, first, middle, last, pimped_bert which corresponds to our CBVFT method). </br>
`--base_model_ckpt`: checkpoint for the weights of the base model that is put in parallel when we use CBVFT.</br>
`--load_path_ckpt`: checkpoint for the weights of the model used for the training (it can be CBVFT or classical BERT model type of weights).</br>
`--batch_size`: batch sizes for the train and val dataloaders.</br>
`--device`: device on which to do the training/testing.</br>
`--eval_only`: 1 to just evaluate the model, 0 to do the training.</br>
`--n_train`: number of training samples per class (5 class in sentiment analysis, 2 in hate speech detection).</br>
`--n_val`: number of validation samples per class. </br>
`--n_epochs`: number of epochs for the training. One epoch goes through all the training samples. </br>
`--lr`: learning rate. </br>
`--val_freq`: validation will be done every val_freq steps during the training. </br>

### Command example
Here is an example of a command to run a training for sentiment analysis:
```
python src/main.py --model bert-med --train_percentages 80,20 --val_percentages 80,20 --train_dataset amazon_books --val_dataset amazon_books,amazon_video --device cuda --mode all,last,first,middle --n_train 5000 --n_val 100
```

And one for hate speech detection:
```
python src/main.py --model bert-med --train_percentages 80,20 --val_percentages 80,20 --train_dataset civil_comments,tweet_eval --val_dataset civil_comments,tweet_eval --device cuda --mode all,last,first,middle --n_train 5000 --n_val 100
```

### Saving experiments
All experiments are saved in a folder describing the command line arguments. Both the best checkpoint (best for accuracy) and the last checkpoint are saved, as well as tensorboard logs to visualize the training.
