# ssn_spinal
Application of spinal net on the sub spectral network for acoustic scene classification

The code contains three programs. 

1. mod_dataset.py : Loads the training and test numpy files. Splits the train into train and val with val ratio as 0.3. Also, trims the spectrograms from [40,501] to [40,500].                         Saves the dataset as x_train.npy, x_val.npy, x_test.npy, y_train.npy, y_val.npy and y_test.npy.

2. dcase_cnn.py : This program runs the baseline dcase model and the baseline_spinalnet model. This program requires the datasets saved from the mod_dataset.py program.

3. ssn_spinal.py : This program runs the basic ssn and ssn spinalnet version. This program requires the datasets saved from the mod_dataset.py program.
