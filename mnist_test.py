# 隨機檢驗十筆mnist中的data
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import load_model
import random as r

(train_feature, train_label), (test_feature, test_label)=mnist.load_data()
test_feature_vector = test_feature.reshape(len(test_feature), 784).astype('float32')
test_feature_normalize = test_feature_vector/255

model = load_model('mnist_mlp_model.h5')

index=[]
for i in range(0, 10):
    a = str(r.randint(1, 10000))
    index.append(a)
print(index)

def show_images_labels_predictions(images, labels, predictions):
    plt.gcf().set_size_inches(12, 14)
    for s in range(0, 10):
        start_id = int(index[s])
        ax = plt.subplot(5, 5, s+1)
        ax.imshow(images[start_id], cmap='binary')
        if( len(predictions)>0 ) :
            title = 'ai='+str(predictions[start_id])
            title += ('(o)'if predictions[start_id]==labels[start_id] else '(x)')
            title += '\nlabel='+ str(labels[start_id])
        else :
            title = 'label='+str(labels[start_id])
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]);ax.set_yticks([])
    plt.show()

prediction = model.predict_classes(test_feature_normalize)
show_images_labels_predictions(test_feature, test_label, prediction)