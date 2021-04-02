import six
import sys
sys.modules['sklearn.externals.six'] = six

from seqlearn.perceptron import StructuredPerceptron



from dataset_manager import *
from predict import *

train_dataset, test_dataset = get_dataset()
X_train, Y_train = get_X_Y(train_dataset)
X_test, Y_test = get_X_Y(test_dataset)




print("debug:", len(X_train)//17)
lengths_train = [len(X_train)]

clf = StructuredPerceptron()
clf.fit(X_train, Y_train, lengths_train)


from seqlearn.evaluation import bio_f_score
lengths_test = [len(X_test)]
print("debug test:", len(X_test))
y_pred = clf.predict(X_test, lengths_test)
print(bio_f_score(Y_test, y_pred))

print(y_pred)
#print(Y_test)
