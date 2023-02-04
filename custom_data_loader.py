import numpy as np
import keras
#list_IDs is just a os.listdir  list from names distributed very well 
#the name is an indicator for the label we will prepare it list from numbers'labels'  distributed very well 
from sklearn.utils import shuffle
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n = 0
        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1
        
        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0
        
        return data 

    def __len__(self):
        'Denotes the number of batches per epoch'
        #print('i am in len ')
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #print('i am in __getitem__ ',index)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print('indexes',indexes)

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #print('list_IDs_temp',list_IDs_temp)

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    

    def __data_generation(self, list_IDs_temp):
        #print('i am in __data_generation ')
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #print(list_IDs_temp)
        X = np.empty((self.batch_size, 261,260, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        #dirr='/content/gdrive/MyDrive/data/'
        #print(list_IDs_temp)
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #print(ID)
            
            if ID[44]=='3':
                
                 lab_=0
            if ID[44]=='4':
                    
                 lab_=1
            if ID[44]=='5':
                
                 lab_=2
            
            
            
            



            #dat_ =(np.abs(np.load(ID)))
            csidata = csiread.Intel(ID,3,1)
            csidata.read()
            dat_ = csidata.get_scaled_csi()
            dat_=np.abs(dat_)
            dat_ =dat_[0:2262]
            
            #print(dat_.shape,(dirr + ID),lab_)

            dat_ =dat_.reshape(261,260,3)
            #print(dat_.shape)

            
            
            
            #dat_ =np.transpose(dat_,(0,2,1))
        
            a_min=dat_.min()
            a_max=dat_.max()
            #dat_=(dat_-a_min)/(a_max-a_min)
            #print(a_min,a_max,ID)
            
            X[i,] = dat_
            #X[i,]=X[i]
            # Store class
            #print('ID',int(ID[2]))
            y[i,]=lab_
            

            #y[i] = self.labels[i]
        #print('\n',keras.utils.to_categorical(y, num_classes=self.n_classes))
        #print('\n',X.max())
        
        return X,keras.utils.to_categorical(y, num_classes=self.n_classes)
        #return X,[X,keras.utils.to_categorical(y, num_classes=self.n_classes)]