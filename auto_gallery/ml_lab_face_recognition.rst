.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_gallery_ml_lab_face_recognition.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_gallery_ml_lab_face_recognition.py:


Lab: Faces recognition using various learning models
====================================================

This lab is inspired by a scikit-learn lab:
`Faces recognition example using eigenfaces and SVMs <https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html>`_

It uses scikit-learan and pytorch models using `skorch <https://github.com/skorch-dev/skorch>`_
(`slides <https://fr.slideshare.net/ThomasFan6/pydata-dc-2018-skorch-a-union-of-scikitlearn-and-pytorch>`_).
  * skorch provides scikit-learn compatible neural network library that wraps PyTorch.
  * skorch abstracts away the training loop, making a lot of boilerplate code obsolete.
    A simple `net.fit(X, y)` is enough.

Note that more sofisticated models can be used,
`see <https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78>`_
for a overview.

Models:

- Eigenfaces unsupervized exploratory analysis.
- `LogisticRegression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_ with L2 regularization (includes model selection with 5CV`_
- `SVM-RBF <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_  (includes model selection with 5CV.
- `MLP using sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>`_ using sklearn (includes model selection with 5CV)
- `MLP using skorch classifier <https://skorch.readthedocs.io/en/stable/classifier.html>`_
- Basic Convnet (ResNet18) using skorch.
- Pretrained ResNet18 using skorch.

Pipelines:

- Univariate feature filtering (Anova) with Logistic-L2
- PCA with LogisticRegression with L2 regularization


.. code-block:: default


    import numpy as np
    from time import time
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    # Preprocesing
    from sklearn import preprocessing
    from sklearn.pipeline import make_pipeline

    from sklearn.datasets import fetch_lfw_people

    # Models
    from sklearn.decomposition import PCA
    import sklearn.linear_model as lm
    import sklearn.svm as svm
    from sklearn.neural_network import MLPClassifier
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.ensemble import GradientBoostingClassifier

    # For pipelines
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    # Pytorch Models
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from skorch import NeuralNetClassifier
    import skorch

    # Use [skorch](https://github.com/skorch-dev/skorch). Install:
    # `conda install -c conda-forge skorch`

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')









Utils
-----


.. code-block:: default



    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Plot a gallery of portraits."""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(min(images.shape[0], n_row * n_col)):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())


    def title(y_pred, y_test, target_names, i):
        """Plot the result of the prediction on a portion of the test set."""
        pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
        true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
        return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


    def label_proportion(x, decimals=2):
        """Labels's proportions."""
        unique, counts = np.unique(x, return_counts=True)
        return dict(zip(unique, np.round(counts / len(x), decimals)))









Download the data
-----------------


.. code-block:: default


    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape

    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total dataset size:
    n_samples: 1288
    n_features: 1850
    n_classes: 7




Split into a training and testing set in stratified way
-------------------------------------------------------


.. code-block:: default


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y)

    print({target_names[lab]: prop for lab, prop in
           label_proportion(y_train).items()})






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    {'Ariel Sharon': 0.06, 'Colin Powell': 0.18, 'Donald Rumsfeld': 0.09, 'George W Bush': 0.41, 'Gerhard Schroeder': 0.08, 'Hugo Chavez': 0.05, 'Tony Blair': 0.11}




Plot mean faces and 4 samples of each individual


.. code-block:: default


    single_faces = [X_train[y_train == lab][:5] for lab in np.unique(y_train)]
    single_faces = np.vstack(single_faces).reshape((5 * n_classes, h, w))

    mean_faces = [X_train[y_train == lab].mean(axis=0) for lab in
                  np.unique(y_train)]
    mean_faces = np.vstack(mean_faces).reshape((n_classes, h, w))

    single_faces[::5, :, :] = mean_faces
    titles = [n for name in target_names for n in [name] * 5]
    plot_gallery(single_faces, titles, h, w, n_row=n_classes, n_col=5)





.. image:: /auto_gallery/images/sphx_glr_ml_lab_face_recognition_001.png
    :class: sphx-glr-single-img





Eigenfaces
----------

Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
dataset): unsupervised feature extraction / dimensionality reduction


.. code-block:: default


    n_components = 150

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Extracting the top 150 eigenfaces from 966 faces
    done in 0.103s
    Projecting the input data on the eigenfaces orthonormal basis




Plot eigenfaces:


.. code-block:: default


    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)





.. image:: /auto_gallery/images/sphx_glr_ml_lab_face_recognition_002.png
    :class: sphx-glr-single-img





LogisticRegression with L2 penalty (with CV-based model selection)
------------------------------------------------------------------

Our goal is to obtain a good balanced accuracy, ie, the macro average
(`macro avg`) of classes' reccalls. In this perspective, the good practices
are:

- Scale input features using either `StandardScaler()` or `MinMaxScaler()`
  "It doesn't harm".
- Re-balance classes' contributions `class_weight='balanced'`
- Do not include an intercept (`fit_intercept=False`) in the model.
  This should reduce the global accuracy `weighted avg`. But rememember that
  we decided to maximize the balanced accuracy.


.. code-block:: default


    lrl2_cv = make_pipeline(
        preprocessing.StandardScaler(),
        # preprocessing.MinMaxScaler(),  # Would have done the job either
        GridSearchCV(lm.LogisticRegression(max_iter=1000, class_weight='balanced',
                                           fit_intercept=False),
                     {'C': 10. ** np.arange(-3, 3)},
                     cv=5, n_jobs=5))

    t0 = time()
    lrl2_cv.fit(X=X_train, y=y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best params found by grid search:")
    print(lrl2_cv.steps[-1][1].best_params_)

    y_pred = lrl2_cv.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    done in 7.246s
    Best params found by grid search:
    {'C': 1.0}
                       precision    recall  f1-score   support

         Ariel Sharon       0.59      0.89      0.71        19
         Colin Powell       0.91      0.83      0.87        59
      Donald Rumsfeld       0.71      0.80      0.75        30
        George W Bush       0.91      0.80      0.86       133
    Gerhard Schroeder       0.70      0.78      0.74        27
          Hugo Chavez       0.62      0.56      0.59        18
           Tony Blair       0.71      0.83      0.77        36

             accuracy                           0.80       322
            macro avg       0.74      0.79      0.75       322
         weighted avg       0.82      0.80      0.80       322

    [[ 17   0   1   0   0   1   0]
     [  2  49   3   3   0   0   2]
     [  3   0  24   1   0   1   1]
     [  7   3   4 107   5   3   4]
     [  0   0   1   0  21   1   4]
     [  0   2   0   3   2  10   1]
     [  0   0   1   3   2   0  30]]




Coeficients


.. code-block:: default


    coefs = lrl2_cv.steps[-1][1].best_estimator_.coef_
    coefs = coefs.reshape(-1, h, w)
    plot_gallery(coefs, target_names, h, w)





.. image:: /auto_gallery/images/sphx_glr_ml_lab_face_recognition_003.png
    :class: sphx-glr-single-img





SVM (with CV-based model selection)
-----------------------------------

Remarks:
- RBF generally requires "large" C (>1)
- Poly generally requires "small" C (<1)


.. code-block:: default


    svm_cv = make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(svm.SVC(class_weight='balanced'),
                     {'kernel': ['poly', 'rbf'], 'C': 10. ** np.arange(-2, 3)},
                     # {'kernel': ['rbf'], 'C': 10. ** np.arange(-1, 4)},
                     cv=5, n_jobs=5))

    t0 = time()
    svm_cv.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best params found by grid search:")
    print(svm_cv.steps[-1][1].best_params_)

    y_pred = svm_cv.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    done in 26.589s
    Best params found by grid search:
    {'C': 0.1, 'kernel': 'poly'}
                       precision    recall  f1-score   support

         Ariel Sharon       0.71      0.89      0.79        19
         Colin Powell       0.84      0.88      0.86        59
      Donald Rumsfeld       0.76      0.87      0.81        30
        George W Bush       0.90      0.89      0.90       133
    Gerhard Schroeder       0.77      0.74      0.75        27
          Hugo Chavez       0.90      0.50      0.64        18
           Tony Blair       0.82      0.78      0.80        36

             accuracy                           0.84       322
            macro avg       0.82      0.79      0.79       322
         weighted avg       0.85      0.84      0.84       322





MLP with sklearn and CV-based model selection
---------------------------------------------

Default parameters:
- alpha, default=0.0001 L2 penalty (regularization term) parameter.
- batch_size=min(200, n_samples)
- learning_rate_init = 0.001 (the important one since we uses adam)
- solver default='adam'
  * sgd: momentum=0.9
  * adam: beta_1, beta_2 default=0.9, 0.999 Exponential decay rates for
    the first and second moment.
- L2 penalty (regularization term) parameter, `alpha` default=0.0001
- tol, default=1e-4


.. code-block:: default


    mlp_param_grid = {"hidden_layer_sizes":
                      # Configurations with 1 hidden layer:
                      [(100, ), (50, ), (25, ), (10, ), (5, ),
                       # Configurations with 2 hidden layers:
                       (100, 50, ), (50, 25, ), (25, 10, ), (10, 5, ),
                       # Configurations with 3 hidden layers:
                       (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, )],
                      "activation": ["relu"], "solver": ["adam"], 'alpha': [0.0001]}

    mlp_cv = make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(estimator=MLPClassifier(random_state=1, max_iter=400),
                     param_grid=mlp_param_grid,
                     cv=5, n_jobs=5))

    t0 = time()
    mlp_cv.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best params found by grid search:")
    print(mlp_cv.steps[-1][1].best_params_)

    y_pred = mlp_cv.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    done in 132.523s
    Best params found by grid search:
    {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'solver': 'adam'}
                       precision    recall  f1-score   support

         Ariel Sharon       0.76      0.84      0.80        19
         Colin Powell       0.88      0.85      0.86        59
      Donald Rumsfeld       0.68      0.77      0.72        30
        George W Bush       0.89      0.91      0.90       133
    Gerhard Schroeder       0.77      0.74      0.75        27
          Hugo Chavez       1.00      0.39      0.56        18
           Tony Blair       0.73      0.83      0.78        36

             accuracy                           0.83       322
            macro avg       0.82      0.76      0.77       322
         weighted avg       0.84      0.83      0.83       322





MLP with pytorch and no model selection
---------------------------------------



.. code-block:: default


    class SimpleMLPClassifierPytorch(nn.Module):
        """Simple (one hidden layer) MLP Classifier with Pytorch."""

        def __init__(self):
            super(SimpleMLPClassifierPytorch, self).__init__()

            self.dense0 = nn.Linear(1850, 100)
            self.nonlin = nn.ReLU()
            self.output = nn.Linear(100, 7)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, X, **kwargs):
            X = self.nonlin(self.dense0(X))
            X = self.softmax(self.output(X))
            return X


    mlp = NeuralNetClassifier(  # Match the parameters with sklearn
        SimpleMLPClassifierPytorch,
        criterion=torch.nn.NLLLoss,
        max_epochs=100,
        batch_size=200,
        optimizer=torch.optim.Adam,
        # optimizer=torch.optim.SGD,
        optimizer__lr=0.001,
        optimizer__betas=(0.9, 0.999),
        optimizer__eps=1e-4,
        optimizer__weight_decay=0.0001,  # L2 regularization
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        device=device,
        verbose=0)

    scaler = preprocessing.MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    t0 = time()
    mlp.fit(X_train_s, y_train)
    print("done in %0.3fs" % (time() - t0))

    y_pred = mlp.predict(X_test_s)
    print(classification_report(y_test, y_pred, target_names=target_names))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    done in 1.927s
                       precision    recall  f1-score   support

         Ariel Sharon       0.78      0.74      0.76        19
         Colin Powell       0.77      0.86      0.82        59
      Donald Rumsfeld       0.68      0.70      0.69        30
        George W Bush       0.89      0.88      0.88       133
    Gerhard Schroeder       0.57      0.63      0.60        27
          Hugo Chavez       1.00      0.33      0.50        18
           Tony Blair       0.74      0.81      0.77        36

             accuracy                           0.79       322
            macro avg       0.77      0.71      0.72       322
         weighted avg       0.80      0.79      0.79       322





Univariate feature filtering (Anova) with Logistic-L2
-----------------------------------------------------


.. code-block:: default


    anova_l2lr = Pipeline([
        ('standardscaler', preprocessing.StandardScaler()),
        ('anova', SelectKBest(f_classif)),
        ('l2lr', lm.LogisticRegression(max_iter=1000, class_weight='balanced',
                                       fit_intercept=False))
    ])

    param_grid = {'anova__k': [50, 100, 500, 1000, 1500, X_train.shape[1]],
                  'l2lr__C': 10. ** np.arange(-3, 3)}
    anova_l2lr_cv = GridSearchCV(anova_l2lr, cv=5,  param_grid=param_grid,
                                 n_jobs=5)

    t0 = time()
    anova_l2lr_cv.fit(X=X_train, y=y_train)
    print("done in %0.3fs" % (time() - t0))

    print("Best params found by grid search:")
    print(anova_l2lr_cv.best_params_)

    y_pred = anova_l2lr_cv.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    done in 18.521s
    Best params found by grid search:
    {'anova__k': 1850, 'l2lr__C': 100.0}
                       precision    recall  f1-score   support

         Ariel Sharon       0.59      0.89      0.71        19
         Colin Powell       0.89      0.83      0.86        59
      Donald Rumsfeld       0.71      0.80      0.75        30
        George W Bush       0.91      0.80      0.85       133
    Gerhard Schroeder       0.72      0.78      0.75        27
          Hugo Chavez       0.62      0.56      0.59        18
           Tony Blair       0.71      0.81      0.75        36

             accuracy                           0.80       322
            macro avg       0.74      0.78      0.75       322
         weighted avg       0.81      0.80      0.80       322





PCA with LogisticRegression with L2 regularization
--------------------------------------------------


.. code-block:: default


    pca_lrl2_cv = make_pipeline(
        PCA(n_components=150, svd_solver='randomized', whiten=True),
        GridSearchCV(lm.LogisticRegression(max_iter=1000, class_weight='balanced',
                                           fit_intercept=False),
                     {'C': 10. ** np.arange(-3, 3)},
                     cv=5, n_jobs=5))

    t0 = time()
    pca_lrl2_cv.fit(X=X_train, y=y_train)
    print("done in %0.3fs" % (time() - t0))

    print("Best params found by grid search:")
    print(pca_lrl2_cv.steps[-1][1].best_params_)

    y_pred = pca_lrl2_cv.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    done in 0.320s
    Best params found by grid search:
    {'C': 0.1}
                       precision    recall  f1-score   support

         Ariel Sharon       0.57      0.89      0.69        19
         Colin Powell       0.85      0.75      0.79        59
      Donald Rumsfeld       0.60      0.80      0.69        30
        George W Bush       0.95      0.69      0.80       133
    Gerhard Schroeder       0.59      0.81      0.69        27
          Hugo Chavez       0.46      0.67      0.55        18
           Tony Blair       0.70      0.78      0.74        36

             accuracy                           0.74       322
            macro avg       0.67      0.77      0.71       322
         weighted avg       0.79      0.74      0.75       322

    [[17  0  1  0  0  1  0]
     [ 5 44  3  3  0  2  2]
     [ 1  2 24  0  0  2  1]
     [ 6  5 10 92  7  9  4]
     [ 0  0  0  0 22  0  5]
     [ 0  1  1  1  3 12  0]
     [ 1  0  1  1  5  0 28]]




Basic ConvNet
-------------

Note that to simplify, do not use pipeline (scaler + CNN) here.
But it would have been simple to do so, since pytorch is warpped in skorch
object that is compatible with sklearn.

Sources:

- `ConvNet on MNIST <https://github.com/skorch-dev/skorch/blob/master/notebooks/MNIST.ipynb>`_
- `NeuralNetClassifier <https://skorch.readthedocs.io/en/stable/classifier.html>`_


.. code-block:: default



    class Cnn(nn.Module):
        """Basic ConvNet Conv(1, 32, 64) -> FC(100, 7) -> softmax."""

        def __init__(self, dropout=0.5, fc_size=4928, n_outputs=7, debug=False):
            super(Cnn, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.conv2_drop = nn.Dropout2d(p=dropout)
            self.fc1 = nn.Linear(fc_size, 100)
            self.fc2 = nn.Linear(100, n_outputs)
            self.fc1_drop = nn.Dropout(p=dropout)
            self.debug = debug

        def forward(self, x):
            x = torch.relu(F.max_pool2d(self.conv1(x), 2))
            x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

            # flatten over channel, height and width = 1600
            x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

            if self.debug:  # trick to get the size of the first FC
                print("### DEBUG: Shape of last convnet=", x.shape,
                      ". FC size=", np.prod(x.shape[1:]))

            x = torch.relu(self.fc1_drop(self.fc1(x)))
            x = torch.softmax(self.fc2(x), dim=-1)
            return x


    torch.manual_seed(0)
    cnn = NeuralNetClassifier(
            Cnn,
            max_epochs=100,
            lr=0.001,
            optimizer=torch.optim.Adam,
            device=device,
            train_split=skorch.dataset.CVSplit(cv=5, stratified=True),
            verbose=0)

    scaler = preprocessing.MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train).reshape(-1, 1, h, w)
    X_test_s = scaler.transform(X_test).reshape(-1, 1, h, w)

    t0 = time()
    cnn.fit(X_train_s, y_train)
    print("done in %0.3fs" % (time() - t0))

    y_pred = cnn.predict(X_test_s)
    print(classification_report(y_test, y_pred, target_names=target_names))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    done in 97.154s
                       precision    recall  f1-score   support

         Ariel Sharon       0.70      0.84      0.76        19
         Colin Powell       0.92      0.92      0.92        59
      Donald Rumsfeld       0.85      0.73      0.79        30
        George W Bush       0.86      0.96      0.91       133
    Gerhard Schroeder       0.88      0.78      0.82        27
          Hugo Chavez       0.90      0.50      0.64        18
           Tony Blair       0.90      0.78      0.84        36

             accuracy                           0.86       322
            macro avg       0.86      0.79      0.81       322
         weighted avg       0.87      0.86      0.86       322





ConvNet with Resnet18
---------------------




.. code-block:: default


    class Resnet18(nn.Module):
        """ResNet 18, pretrained, with one input chanel and 7 outputs."""

        def __init__(self, in_channels=1, n_outputs=7):
            super(Resnet18, self).__init__()

            # self.model = torchvision.models.resnet18()
            self.model = torchvision.models.resnet18(pretrained=True)

            # original definition of the first layer on the renset class
            # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            #                        bias=False)
            # one channel input (greyscale):
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                         padding=3, bias=False)

            # Last layer
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, n_outputs)

        def forward(self, x):
            return self.model(x)


    torch.manual_seed(0)
    resnet = NeuralNetClassifier(
        Resnet18,
        # `CrossEntropyLoss` combines `LogSoftmax and `NLLLoss`
        criterion=nn.CrossEntropyLoss,
        max_epochs=50,
        batch_size=128,  # default value
        optimizer=torch.optim.Adam,
        # optimizer=torch.optim.SGD,
        optimizer__lr=0.001,
        optimizer__betas=(0.9, 0.999),
        optimizer__eps=1e-4,
        optimizer__weight_decay=0.0001,  # L2 regularization
        # Shuffle training data on each epoch
        # iterator_train__shuffle=True,
        train_split=skorch.dataset.CVSplit(cv=5, stratified=True),
        device=device,
        verbose=0)

    scaler = preprocessing.MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train).reshape(-1, 1, h, w)
    X_test_s = scaler.transform(X_test).reshape(-1, 1, h, w)

    t0 = time()
    resnet.fit(X_train_s, y_train)
    print("done in %0.3fs" % (time() - t0))

    # Continue training a model (warm re-start):
    # resnet.partial_fit(X_train_s, y_train)

    y_pred = resnet.predict(X_test_s)
    print(classification_report(y_test, y_pred, target_names=target_names))

    epochs = np.arange(len(resnet.history[:, 'train_loss'])) + 1
    plt.plot(epochs, resnet.history[:, 'train_loss'], '-b', label='train_loss')
    plt.plot(epochs, resnet.history[:, 'valid_loss'], '-r', label='valid_loss')
    plt.plot(epochs, resnet.history[:, 'valid_acc'], '--r', label='valid_acc')
    plt.legend()
    plt.show()



.. image:: /auto_gallery/images/sphx_glr_ml_lab_face_recognition_004.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    done in 462.056s
                       precision    recall  f1-score   support

         Ariel Sharon       0.90      0.95      0.92        19
         Colin Powell       0.98      0.93      0.96        59
      Donald Rumsfeld       0.73      0.80      0.76        30
        George W Bush       0.88      0.98      0.93       133
    Gerhard Schroeder       0.77      0.63      0.69        27
          Hugo Chavez       0.90      0.50      0.64        18
           Tony Blair       0.88      0.83      0.86        36

             accuracy                           0.88       322
            macro avg       0.86      0.80      0.82       322
         weighted avg       0.88      0.88      0.87       322

    /home/ed203246/git/pystatsml/labs/ml_lab_face_recognition.py:528: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 12 minutes  29.858 seconds)


.. _sphx_glr_download_auto_gallery_ml_lab_face_recognition.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: ml_lab_face_recognition.py <ml_lab_face_recognition.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: ml_lab_face_recognition.ipynb <ml_lab_face_recognition.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
