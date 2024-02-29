import bentoml

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

iris = load_iris()

x=iris.data[:, :4]
y=iris.target
model.fit(x,y)

bento_model = bentoml.sklearn.save_model('Kneighbors',model)
print(f"Model saved : {bento_model}")