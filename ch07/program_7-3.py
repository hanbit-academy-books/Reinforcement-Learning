import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=ds.mnist.load_data() # data preparation
xx_train=x_train.reshape(60000,28,28,1)
xx_test=x_test.reshape(10000,28,28,1)
xx_train=xx_train.astype(np.float32)/255.0
xx_test=xx_test.astype(np.float32)/255.0
yy_train=tf.keras.utils.to_categorical(y_train,10)
yy_test=tf.keras.utils.to_categorical(y_test,10)

cnn=Sequential()
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())
cnn.add(Dense(units=512,activation='relu'))
cnn.add(Dense(units=10,activation='softmax'))

cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
hist=cnn.fit(xx_train,yy_train,batch_size=128,epochs=50,validation_data=(xx_test,yy_test),verbose=2)

res=cnn.evaluate(xx_test,yy_test,verbose=0) # accuracy for test set
print('accuracy=',res[1]*100)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()
