import numpy as np
from keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

(train_feature, train_label), (test_feature, test_label) = mnist.load_data()
train_feature_vector = train_feature.reshape(len(train_feature), 784).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature), 784).astype('float32')
train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255
train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)

model = Sequential()
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=train_feature_normalize, y=train_label_onehot, validation_split=0.1, epochs=10, batch_size=20, verbose=2)
scores = model.evaluate(test_feature_normalize, test_label_onehot)

print('準確率=', scores[1])

model.save('Mnist_mlp_model.h5')
print('save as Mnist_mlp_model.h5')

# 原作 validation_split=0.2, epochs=10, batch_size=200, ：
#   1.準確率= 0.9764000177383423
#   2.準確率= 0.9758999943733215
#   3.準確率= 0.9757000207901001
#   4.準確率= 0.9760000109672546
#   5.準確率= 0.9764999747276306
#   6.準確率= 0.9768000245094299
#   7.準確率= 0.9761999845504761
#   8.準確率= 0.9761999845504761
#   9.準確率= 0.9758999943733215
#   10.準確率= 0.9760000109672546

# epoch=100, 準確率= 0.9822999835014343
# batch_size=2000, 準確率= 0.9549000263214111
# batch_size=20, 準確率= 0.9811000227928162
# validation_split=0.1, 準確率= 0.9804999828338623
# validation_split=0.1, epochs=50, batch_size=20, 準確率= 0.9797999858856201
# validation_split=0.1, epochs=10, batch_size=20, 準確率= 0.98089998960495 存為'Mnist_mlp_model.h5'
