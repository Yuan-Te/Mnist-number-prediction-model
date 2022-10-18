# 隨機顯示十筆mnist中的data
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random as r
(train_feature, train_label), (test_feature, test_label)=mnist.load_data()
index=[]
for i in range(0, 10):
    a = str(r.randint(1, 60000))
    index.append(a)
print(index)

def show_images_labels_predictions(images, labels):
    plt.gcf().set_size_inches(12, 14)
    for s in range(10):
        start_id = int(index[s])
        ax = plt.subplot(5, 5, s+1)
        ax.imshow(images[start_id], cmap='binary')
        title = 'label='+str(labels[start_id])
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]);ax.set_yticks([])
    plt.show()

show_images_labels_predictions(train_feature,train_label)

