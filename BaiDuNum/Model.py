from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras.models import load_model

import pandas as pd
import numpy as np
import matplotlib.image as mpimg
x_train=np.ndarray((100000,1,40,120))
label=pd.read_csv('E:\\PycharmProj\\BaiDuNum\\label_dummies.csv')
y_train=np.array(label[label.columns[range(10,99)]])
for i in range(100000):
    path='E:\\image_contest_level_1\\gray\\small\\zero\\'+str(i)+'.png'
    img=mpimg.imread(path)
    x_train[i]=img

x_train=x_train.reshape((100000,40,120,1))

y_train=y_train[0:100000]

optimizer = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'

losses=[]
val_losses=[]
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        print(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

earlystop=EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
history=LossHistory()
# 建造模型
model = Sequential()

model.add(Conv2D(32, (3, 3), border_mode='same', input_shape=(40,120,1), activation='relu'))
model.add(Conv2D(32, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), border_mode='same', activation='relu'))
model.add(Conv2D(64, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), border_mode='same', activation='relu'))
model.add(Conv2D(128, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), border_mode='same', activation='relu'))
model.add(Conv2D(256, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(89))
model.add(Activation('softmax'))

model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

#model=load_model('model.h5')
model.fit(x_train,y_train,batch_size=200,epochs=200,validation_split=0.2,callbacks=[history,earlystop])

model.save('model3.h5')
file=open('loss_result.txt','w')
file.write(str(losses))
file.write(str(val_losses))
file.close()