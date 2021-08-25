Instructions to run test.py: 
    1) Download the 'trained_model.pt' file from the OneDrive link below (~1.5GB).
    2) Place 'trained_model.pt' in a folder that also contains the numpy array of images to test the model with
    3) Change the "FILE_PATH" variable in test.py on line 29 to the path of the folder created in (2)
    4) If the the numpy array of images has a different name than "Images.npy", the "Images" variable can be modified on line 160. Replace the "Images.npy" to the name of your image numpy array
    5) The test_model function runs the pretrained model on the given image data. The outputs are printed to the console as well as placed in a variable called "predicted" on line 164

Link to trained model parameters 'trained_model.pt' file required for test.py: 
https://uflorida-my.sharepoint.com/:u:/g/personal/wespiard_ufl_edu/EZW5LLX63dpLt-DRITW-0UwB0sP2avtf-HwWAFYlB9zvWQ?e=bocj6X\


Instructions to modify train.py:
    1) The main training function is "train_model" at line 117.
    2) Change the variable 'file_path' at line 356 should be the path to a folder containing a numpy array called Images.npy and a label array called Labels.npy. Change the path name on your local machine to train on the Images.npy array.
    3) If the Image np array or Label np array have different naming conventions, these names can be changed in lines 358 and 359 in the "Images" and "Labels" variables. 
    4) To modify any of the parameters used for training, lines 177-190 define the variables for number of epochs to train, batch size, learning rate, optimizer, and loss function
    
