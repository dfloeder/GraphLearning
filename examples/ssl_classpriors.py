#import graphlearning as gl
import sys
print(sys.path)
#sys.path.remove('/home/dfloeder/.local/lib/python3.8/site-packages/graphlearning/__init__.py')

#sys.path.append('home/dfloeder/Spring22/Thesis/GraphLearning/graphlearning')
#sys.path.remove('/home/dfloeder/.local/lib/python3.8/site-packages/graphlearning-1.0.9-py3.8-linux-x86_64.egg')
#print(sys.path)

import graphlearning as gl
print(gl.__file__)

labels = gl.datasets.load('mnist', labels_only=True)
W = gl.weightmatrix.knn('mnist', 10, metric='vae')

num_train_per_class = 1
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

class_priors = gl.utils.class_priors(labels)
model = gl.ssl.laplace(W, class_priors=class_priors)
model.fit(train_ind,train_labels)

pred_labels = model.predict(ignore_class_priors=True)
accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
print(model.name + ' without class priors: %.2f%%'%accuracy)

pred_labels = model.predict()
accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
print(model.name + ' with class priors: %.2f%%'%accuracy)


