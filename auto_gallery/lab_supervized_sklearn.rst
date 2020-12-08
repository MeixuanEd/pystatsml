.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_gallery_lab_supervized_sklearn.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_gallery_lab_supervized_sklearn.py:


Created on Mon Nov 23 00:48:03 2020

@author: ed203246




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    # Tuning hyper-parameters for precision

    Best parameters set found on development set:

    {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

    Grid scores on development set:

    0.974 (+/-0.031) for {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}
    0.943 (+/-0.020) for {'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'}
    0.981 (+/-0.024) for {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
    0.969 (+/-0.025) for {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}
    0.981 (+/-0.024) for {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
    0.969 (+/-0.031) for {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}
    0.981 (+/-0.024) for {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
    0.969 (+/-0.031) for {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'}
    0.959 (+/-0.044) for {'C': 1, 'kernel': 'linear'}
    0.959 (+/-0.044) for {'C': 10, 'kernel': 'linear'}
    0.959 (+/-0.044) for {'C': 100, 'kernel': 'linear'}
    0.959 (+/-0.044) for {'C': 1000, 'kernel': 'linear'}

    Detailed classification report:

    The model is trained on the full development set.
    The scores are computed on the full evaluation set.

                  precision    recall  f1-score   support

               0       1.00      0.99      1.00       147
               1       0.97      0.99      0.98       147
               2       1.00      0.99      1.00       138
               3       0.99      0.94      0.96       150
               4       0.99      0.98      0.98       137
               5       0.97      0.96      0.96       153
               6       0.99      0.99      0.99       141
               7       0.96      1.00      0.98       140
               8       0.96      0.92      0.94       146
               9       0.93      0.99      0.96       139

        accuracy                           0.97      1438
       macro avg       0.98      0.98      0.98      1438
    weighted avg       0.98      0.97      0.97      1438


    # Tuning hyper-parameters for recall

    Best parameters set found on development set:

    {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

    Grid scores on development set:

    0.968 (+/-0.037) for {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}
    0.931 (+/-0.028) for {'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'}
    0.976 (+/-0.032) for {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
    0.963 (+/-0.025) for {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}
    0.976 (+/-0.032) for {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
    0.963 (+/-0.030) for {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}
    0.976 (+/-0.032) for {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
    0.963 (+/-0.030) for {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'}
    0.952 (+/-0.043) for {'C': 1, 'kernel': 'linear'}
    0.952 (+/-0.043) for {'C': 10, 'kernel': 'linear'}
    0.952 (+/-0.043) for {'C': 100, 'kernel': 'linear'}
    0.952 (+/-0.043) for {'C': 1000, 'kernel': 'linear'}

    Detailed classification report:

    The model is trained on the full development set.
    The scores are computed on the full evaluation set.

                  precision    recall  f1-score   support

               0       1.00      0.99      1.00       147
               1       0.97      0.99      0.98       147
               2       1.00      0.99      1.00       138
               3       0.99      0.94      0.96       150
               4       0.99      0.98      0.98       137
               5       0.97      0.96      0.96       153
               6       0.99      0.99      0.99       141
               7       0.96      1.00      0.98       140
               8       0.96      0.92      0.94       146
               9       0.93      0.99      0.96       139

        accuracy                           0.97      1438
       macro avg       0.98      0.98      0.98      1438
    weighted avg       0.98      0.97      0.97      1438








|


.. code-block:: default

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC

    print(__doc__)

    # Loading the Digits dataset
    digits = datasets.load_digits()

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    #mask = (y == 0) | (y == 8)
    #mask.sum()
    #X = X[mask]
    #y = y[mask]

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    # Note the problem is too easy: the hyperparameter plateau is too flat and the
    # output model is the same for precision and recall with ties in quality.

.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  1.130 seconds)


.. _sphx_glr_download_auto_gallery_lab_supervized_sklearn.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: lab_supervized_sklearn.py <lab_supervized_sklearn.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: lab_supervized_sklearn.ipynb <lab_supervized_sklearn.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
