.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_gallery_scipy_pandas.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_gallery_scipy_pandas.py:


Pandas: data manipulation
=========================

It is often said that 80% of data analysis is spent on the cleaning and
small, but important, aspect of data manipulation and cleaning with Pandas.

**Sources**:

- Kevin Markham: https://github.com/justmarkham
- Pandas doc: http://pandas.pydata.org/pandas-docs/stable/index.html

**Data structures**

- **Series** is a one-dimensional labeled array capable of holding any data
  type (integers, strings, floating point numbers, Python objects, etc.).
  The axis labels are collectively referred to as the index. The basic method
  to create a Series is to call `pd.Series([1,3,5,np.nan,6,8])`

- **DataFrame** is a 2-dimensional labeled data structure with columns of
  potentially different types. You can think of it like a spreadsheet or SQL
  table, or a dict of Series objects. It stems from the `R data.frame()`
  object.


.. code-block:: default


    import pandas as pd
    import numpy as np








Create DataFrame
----------------


.. code-block:: default


    columns = ['name', 'age', 'gender', 'job']

    user1 = pd.DataFrame([['alice', 19, "F", "student"],
                          ['john', 26, "M", "student"]],
                         columns=columns)

    user2 = pd.DataFrame([['eric', 22, "M", "student"],
                          ['paul', 58, "F", "manager"]],
                         columns=columns)

    user3 = pd.DataFrame(dict(name=['peter', 'julie'],
                              age=[33, 44], gender=['M', 'F'],
                              job=['engineer', 'scientist']))

    print(user3)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

        name  age gender        job
    0  peter   33      M   engineer
    1  julie   44      F  scientist




Combining DataFrames
--------------------

Concatenate DataFrame
~~~~~~~~~~~~~~~~~~~~~


.. code-block:: default


    user1.append(user2)
    users = pd.concat([user1, user2, user3])
    print(users)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

        name  age gender        job
    0  alice   19      F    student
    1   john   26      M    student
    0   eric   22      M    student
    1   paul   58      F    manager
    0  peter   33      M   engineer
    1  julie   44      F  scientist




Join DataFrame
~~~~~~~~~~~~~~


.. code-block:: default


    user4 = pd.DataFrame(dict(name=['alice', 'john', 'eric', 'julie'],
                              height=[165, 180, 175, 171]))
    print(user4)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

        name  height
    0  alice     165
    1   john     180
    2   eric     175
    3  julie     171




Use intersection of keys from both frames


.. code-block:: default


    merge_inter = pd.merge(users, user4)

    print(merge_inter)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

        name  age gender        job  height
    0  alice   19      F    student     165
    1   john   26      M    student     180
    2   eric   22      M    student     175
    3  julie   44      F  scientist     171




Use union of keys from both frames


.. code-block:: default


    users = pd.merge(users, user4, on="name", how='outer')
    print(users)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

        name  age gender        job  height
    0  alice   19      F    student   165.0
    1   john   26      M    student   180.0
    2   eric   22      M    student   175.0
    3   paul   58      F    manager     NaN
    4  peter   33      M   engineer     NaN
    5  julie   44      F  scientist   171.0




Reshaping by pivoting
~~~~~~~~~~~~~~~~~~~~~

“Unpivots” a DataFrame from wide format to long (stacked) format,


.. code-block:: default


    staked = pd.melt(users, id_vars="name", var_name="variable", value_name="value")
    print(staked)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

         name variable      value
    0   alice      age         19
    1    john      age         26
    2    eric      age         22
    3    paul      age         58
    4   peter      age         33
    5   julie      age         44
    6   alice   gender          F
    7    john   gender          M
    8    eric   gender          M
    9    paul   gender          F
    10  peter   gender          M
    11  julie   gender          F
    12  alice      job    student
    13   john      job    student
    14   eric      job    student
    15   paul      job    manager
    16  peter      job   engineer
    17  julie      job  scientist
    18  alice   height        165
    19   john   height        180
    20   eric   height        175
    21   paul   height        NaN
    22  peter   height        NaN
    23  julie   height        171




“pivots” a DataFrame from long (stacked) format to wide format,


.. code-block:: default


    print(staked.pivot(index='name', columns='variable', values='value'))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    variable age gender height        job
    name                                 
    alice     19      F    165    student
    eric      22      M    175    student
    john      26      M    180    student
    julie     44      F    171  scientist
    paul      58      F    NaN    manager
    peter     33      M    NaN   engineer




Summarizing
-----------



.. code-block:: default


    users                   # print the first 30 and last 30 rows
    type(users)             # DataFrame
    users.head()            # print the first 5 rows
    users.tail()            # print the last 5 rows







.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>name</th>
              <th>age</th>
              <th>gender</th>
              <th>job</th>
              <th>height</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>1</th>
              <td>john</td>
              <td>26</td>
              <td>M</td>
              <td>student</td>
              <td>180.0</td>
            </tr>
            <tr>
              <th>2</th>
              <td>eric</td>
              <td>22</td>
              <td>M</td>
              <td>student</td>
              <td>175.0</td>
            </tr>
            <tr>
              <th>3</th>
              <td>paul</td>
              <td>58</td>
              <td>F</td>
              <td>manager</td>
              <td>NaN</td>
            </tr>
            <tr>
              <th>4</th>
              <td>peter</td>
              <td>33</td>
              <td>M</td>
              <td>engineer</td>
              <td>NaN</td>
            </tr>
            <tr>
              <th>5</th>
              <td>julie</td>
              <td>44</td>
              <td>F</td>
              <td>scientist</td>
              <td>171.0</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

Descriptive statistics


.. code-block:: default


    users.describe(include="all")






.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>name</th>
              <th>age</th>
              <th>gender</th>
              <th>job</th>
              <th>height</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>count</th>
              <td>6</td>
              <td>6.000000</td>
              <td>6</td>
              <td>6</td>
              <td>4.000000</td>
            </tr>
            <tr>
              <th>unique</th>
              <td>6</td>
              <td>NaN</td>
              <td>2</td>
              <td>4</td>
              <td>NaN</td>
            </tr>
            <tr>
              <th>top</th>
              <td>peter</td>
              <td>NaN</td>
              <td>M</td>
              <td>student</td>
              <td>NaN</td>
            </tr>
            <tr>
              <th>freq</th>
              <td>1</td>
              <td>NaN</td>
              <td>3</td>
              <td>3</td>
              <td>NaN</td>
            </tr>
            <tr>
              <th>mean</th>
              <td>NaN</td>
              <td>33.666667</td>
              <td>NaN</td>
              <td>NaN</td>
              <td>172.750000</td>
            </tr>
            <tr>
              <th>std</th>
              <td>NaN</td>
              <td>14.895189</td>
              <td>NaN</td>
              <td>NaN</td>
              <td>6.344289</td>
            </tr>
            <tr>
              <th>min</th>
              <td>NaN</td>
              <td>19.000000</td>
              <td>NaN</td>
              <td>NaN</td>
              <td>165.000000</td>
            </tr>
            <tr>
              <th>25%</th>
              <td>NaN</td>
              <td>23.000000</td>
              <td>NaN</td>
              <td>NaN</td>
              <td>169.500000</td>
            </tr>
            <tr>
              <th>50%</th>
              <td>NaN</td>
              <td>29.500000</td>
              <td>NaN</td>
              <td>NaN</td>
              <td>173.000000</td>
            </tr>
            <tr>
              <th>75%</th>
              <td>NaN</td>
              <td>41.250000</td>
              <td>NaN</td>
              <td>NaN</td>
              <td>176.250000</td>
            </tr>
            <tr>
              <th>max</th>
              <td>NaN</td>
              <td>58.000000</td>
              <td>NaN</td>
              <td>NaN</td>
              <td>180.000000</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

Meta-information


.. code-block:: default


    users.index             # "Row names"
    users.columns           # column names
    users.dtypes            # data types of each column
    users.values            # underlying numpy array
    users.shape             # number of rows and columns





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (6, 5)



Columns selection
-----------------


.. code-block:: default


    users['gender']         # select one column
    type(users['gender'])   # Series
    users.gender            # select one column using the DataFrame

    # select multiple columns
    users[['age', 'gender']]        # select two columns
    my_cols = ['age', 'gender']     # or, create a list...
    users[my_cols]                  # ...and use that list to select columns
    type(users[my_cols])            # DataFrame








Rows selection (basic)
----------------------

`iloc` is strictly integer position based


.. code-block:: default


    df = users.copy()
    df.iloc[0]     # first row
    df.iloc[0, :]  # first row
    df.iloc[0, 0]  # first item of first row
    df.iloc[0, 0] = 55








`loc` supports mixed integer and label based access.


.. code-block:: default


    df.loc[0]         # first row
    df.loc[0, :]      # first row
    df.loc[0, "age"]  # age item of first row
    df.loc[0, "age"] = 55








Selection and index

Select females into a new DataFrame


.. code-block:: default


    df = users[users.gender == "F"]
    print(df)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

        name  age gender        job  height
    0  alice   19      F    student   165.0
    3   paul   58      F    manager     NaN
    5  julie   44      F  scientist   171.0




Get the two first rows using `iloc` (strictly integer position)


.. code-block:: default


    df.iloc[[0, 1], :]  # Ok, but watch the index: 0, 3






.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>name</th>
              <th>age</th>
              <th>gender</th>
              <th>job</th>
              <th>height</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>0</th>
              <td>alice</td>
              <td>19</td>
              <td>F</td>
              <td>student</td>
              <td>165.0</td>
            </tr>
            <tr>
              <th>3</th>
              <td>paul</td>
              <td>58</td>
              <td>F</td>
              <td>manager</td>
              <td>NaN</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

Use `loc`


.. code-block:: default


    try:
        df.loc[[0, 1], :]  # Failed
    except KeyError as err:
        print(err)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    "Passing list-likes to .loc or [] with any missing labels is no longer supported. The following labels were missing: Int64Index([1], dtype='int64'). See https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike"




Reset index


.. code-block:: default


    df = df.reset_index(drop=True)  # Watch the index
    print(df)
    print(df.loc[[0, 1], :])






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

        name  age gender        job  height
    0  alice   19      F    student   165.0
    1   paul   58      F    manager     NaN
    2  julie   44      F  scientist   171.0
        name  age gender      job  height
    0  alice   19      F  student   165.0
    1   paul   58      F  manager     NaN




Sorting
-------

Rows iteration
--------------


.. code-block:: default


    df = users[:2].copy()








`iterrows()`: slow, get series, **read-only**

- Returns (index, Series) pairs.
- Slow because iterrows boxes the data into a Series.
- Retrieve fields with column name
- **Don't modify something you are iterating over**. Depending on the data types,
  the iterator returns a copy and not a view, and writing to it will have no
  effect.


.. code-block:: default


    for idx, row in df.iterrows():
        print(row["name"], row["age"])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    alice 19
    john 26




`itertuples()`: fast, get namedtuples, **read-only**

- Returns namedtuples of the values and which is generally faster than iterrows.
- Fast, because itertuples does not box the data into a Series.
- Retrieve fields with integer index starting from 0.
- Names will be renamed to positional names if they are invalid Python
identifier


.. code-block:: default


    for tup in df.itertuples():
        print(tup[1], tup[2])






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    alice 19
    john 26




iter using `loc[i, ...]`: read and **write**


.. code-block:: default


    for i in range(df.shape[0]):
        df.loc[i, "age"] *= 10  # df is modified









Rows selection (filtering)
--------------------------

simple logical filtering on numerical values


.. code-block:: default


    users[users.age < 20]        # only show users with age < 20
    young_bool = users.age < 20  # or, create a Series of booleans...
    young = users[young_bool]            # ...and use that Series to filter rows
    users[users.age < 20].job    # select one column from the filtered results
    print(young)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

        name  age gender      job  height
    0  alice   19      F  student   165.0




simple logical filtering on categorial values


.. code-block:: default


    users[users.job == 'student']
    users[users.job.isin(['student', 'engineer'])]
    users[users['job'].str.contains("stu|scient")]







.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>name</th>
              <th>age</th>
              <th>gender</th>
              <th>job</th>
              <th>height</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>0</th>
              <td>alice</td>
              <td>19</td>
              <td>F</td>
              <td>student</td>
              <td>165.0</td>
            </tr>
            <tr>
              <th>1</th>
              <td>john</td>
              <td>26</td>
              <td>M</td>
              <td>student</td>
              <td>180.0</td>
            </tr>
            <tr>
              <th>2</th>
              <td>eric</td>
              <td>22</td>
              <td>M</td>
              <td>student</td>
              <td>175.0</td>
            </tr>
            <tr>
              <th>5</th>
              <td>julie</td>
              <td>44</td>
              <td>F</td>
              <td>scientist</td>
              <td>171.0</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

Advanced logical filtering


.. code-block:: default


    users[users.age < 20][['age', 'job']]            # select multiple columns
    users[(users.age > 20) & (users.gender == 'M')]  # use multiple conditions







.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>name</th>
              <th>age</th>
              <th>gender</th>
              <th>job</th>
              <th>height</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>1</th>
              <td>john</td>
              <td>26</td>
              <td>M</td>
              <td>student</td>
              <td>180.0</td>
            </tr>
            <tr>
              <th>2</th>
              <td>eric</td>
              <td>22</td>
              <td>M</td>
              <td>student</td>
              <td>175.0</td>
            </tr>
            <tr>
              <th>4</th>
              <td>peter</td>
              <td>33</td>
              <td>M</td>
              <td>engineer</td>
              <td>NaN</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

Sorting
-------


.. code-block:: default


    df = users.copy()

    df.age.sort_values()                      # only works for a Series
    df.sort_values(by='age')                  # sort rows by a specific column
    df.sort_values(by='age', ascending=False) # use descending order instead
    df.sort_values(by=['job', 'age'])         # sort by multiple columns
    df.sort_values(by=['job', 'age'], inplace=True) # modify df

    print(df)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

        name  age gender        job  height
    4  peter   33      M   engineer     NaN
    3   paul   58      F    manager     NaN
    5  julie   44      F  scientist   171.0
    0  alice   19      F    student   165.0
    2   eric   22      M    student   175.0
    1   john   26      M    student   180.0




Descriptive statistics
----------------------

Summarize all numeric columns


.. code-block:: default


    print(df.describe())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

                 age      height
    count   6.000000    4.000000
    mean   33.666667  172.750000
    std    14.895189    6.344289
    min    19.000000  165.000000
    25%    23.000000  169.500000
    50%    29.500000  173.000000
    75%    41.250000  176.250000
    max    58.000000  180.000000




Summarize all columns


.. code-block:: default


    print(df.describe(include='all'))
    print(df.describe(include=['object']))  # limit to one (or more) types





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

            name        age gender      job      height
    count      6   6.000000      6        6    4.000000
    unique     6        NaN      2        4         NaN
    top     john        NaN      M  student         NaN
    freq       1        NaN      3        3         NaN
    mean     NaN  33.666667    NaN      NaN  172.750000
    std      NaN  14.895189    NaN      NaN    6.344289
    min      NaN  19.000000    NaN      NaN  165.000000
    25%      NaN  23.000000    NaN      NaN  169.500000
    50%      NaN  29.500000    NaN      NaN  173.000000
    75%      NaN  41.250000    NaN      NaN  176.250000
    max      NaN  58.000000    NaN      NaN  180.000000
            name gender      job
    count      6      6        6
    unique     6      2        4
    top     john      M  student
    freq       1      3        3




Statistics per group (groupby)


.. code-block:: default


    print(df.groupby("job").mean())

    print(df.groupby("job")["age"].mean())

    print(df.groupby("job").describe(include='all'))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

                     age      height
    job                             
    engineer   33.000000         NaN
    manager    58.000000         NaN
    scientist  44.000000  171.000000
    student    22.333333  173.333333
    job
    engineer     33.000000
    manager      58.000000
    scientist    44.000000
    student      22.333333
    Name: age, dtype: float64
               name                                                         age             ... gender           height                                                                         
              count unique    top freq mean  std  min  25%  50%  75%  max count unique top  ...    50%  75%  max  count unique top freq        mean       std    min    25%    50%    75%    max
    job                                                                                     ...                                                                                                 
    engineer      1      1  peter    1  NaN  NaN  NaN  NaN  NaN  NaN  NaN   1.0    NaN NaN  ...    NaN  NaN  NaN    0.0    NaN NaN  NaN         NaN       NaN    NaN    NaN    NaN    NaN    NaN
    manager       1      1   paul    1  NaN  NaN  NaN  NaN  NaN  NaN  NaN   1.0    NaN NaN  ...    NaN  NaN  NaN    0.0    NaN NaN  NaN         NaN       NaN    NaN    NaN    NaN    NaN    NaN
    scientist     1      1  julie    1  NaN  NaN  NaN  NaN  NaN  NaN  NaN   1.0    NaN NaN  ...    NaN  NaN  NaN    1.0    NaN NaN  NaN  171.000000       NaN  171.0  171.0  171.0  171.0  171.0
    student       3      3  alice    1  NaN  NaN  NaN  NaN  NaN  NaN  NaN   3.0    NaN NaN  ...    NaN  NaN  NaN    3.0    NaN NaN  NaN  173.333333  7.637626  165.0  170.0  175.0  177.5  180.0

    [4 rows x 44 columns]




Groupby in a loop


.. code-block:: default


    for grp, data in df.groupby("job"):
        print(grp, data)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    engineer     name  age gender       job  height
    4  peter   33      M  engineer     NaN
    manager    name  age gender      job  height
    3  paul   58      F  manager     NaN
    scientist     name  age gender        job  height
    5  julie   44      F  scientist   171.0
    student     name  age gender      job  height
    0  alice   19      F  student   165.0
    2   eric   22      M  student   175.0
    1   john   26      M  student   180.0




Quality check
-------------

Remove duplicate data
~~~~~~~~~~~~~~~~~~~~~


.. code-block:: default



    df = users.append(users.iloc[0], ignore_index=True)

    print(df.duplicated())                 # Series of booleans
    # (True if a row is identical to a previous row)
    df.duplicated().sum()                  # count of duplicates
    df[df.duplicated()]                    # only show duplicates
    df.age.duplicated()                    # check a single column for duplicates
    df.duplicated(['age', 'gender']).sum() # specify columns for finding duplicates
    df = df.drop_duplicates()              # drop duplicate rows






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    dtype: bool




Missing data
~~~~~~~~~~~~


.. code-block:: default


    # Missing values are often just excluded
    df = users.copy()

    df.describe(include='all')

    # find missing values in a Series
    df.height.isnull()           # True if NaN, False otherwise
    df.height.notnull()          # False if NaN, True otherwise
    df[df.height.notnull()]      # only show rows where age is not NaN
    df.height.isnull().sum()     # count the missing values

    # find missing values in a DataFrame
    df.isnull()             # DataFrame of booleans
    df.isnull().sum()       # calculate the sum of each column






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    name      0
    age       0
    gender    0
    job       0
    height    2
    dtype: int64



Strategy 1: drop missing values


.. code-block:: default


    df.dropna()             # drop a row if ANY values are missing
    df.dropna(how='all')    # drop a row only if ALL values are missing







.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>name</th>
              <th>age</th>
              <th>gender</th>
              <th>job</th>
              <th>height</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>0</th>
              <td>alice</td>
              <td>19</td>
              <td>F</td>
              <td>student</td>
              <td>165.0</td>
            </tr>
            <tr>
              <th>1</th>
              <td>john</td>
              <td>26</td>
              <td>M</td>
              <td>student</td>
              <td>180.0</td>
            </tr>
            <tr>
              <th>2</th>
              <td>eric</td>
              <td>22</td>
              <td>M</td>
              <td>student</td>
              <td>175.0</td>
            </tr>
            <tr>
              <th>3</th>
              <td>paul</td>
              <td>58</td>
              <td>F</td>
              <td>manager</td>
              <td>NaN</td>
            </tr>
            <tr>
              <th>4</th>
              <td>peter</td>
              <td>33</td>
              <td>M</td>
              <td>engineer</td>
              <td>NaN</td>
            </tr>
            <tr>
              <th>5</th>
              <td>julie</td>
              <td>44</td>
              <td>F</td>
              <td>scientist</td>
              <td>171.0</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

Strategy 2: fill in missing values


.. code-block:: default


    df.height.mean()
    df = users.copy()
    df.loc[df.height.isnull(), "height"] = df["height"].mean()

    print(df)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

        name  age gender        job  height
    0  alice   19      F    student  165.00
    1   john   26      M    student  180.00
    2   eric   22      M    student  175.00
    3   paul   58      F    manager  172.75
    4  peter   33      M   engineer  172.75
    5  julie   44      F  scientist  171.00




Renaming
--------

Rename columns


.. code-block:: default


    df = users.copy()
    df.rename(columns={'name': 'NAME'})






.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>NAME</th>
              <th>age</th>
              <th>gender</th>
              <th>job</th>
              <th>height</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>0</th>
              <td>alice</td>
              <td>19</td>
              <td>F</td>
              <td>student</td>
              <td>165.0</td>
            </tr>
            <tr>
              <th>1</th>
              <td>john</td>
              <td>26</td>
              <td>M</td>
              <td>student</td>
              <td>180.0</td>
            </tr>
            <tr>
              <th>2</th>
              <td>eric</td>
              <td>22</td>
              <td>M</td>
              <td>student</td>
              <td>175.0</td>
            </tr>
            <tr>
              <th>3</th>
              <td>paul</td>
              <td>58</td>
              <td>F</td>
              <td>manager</td>
              <td>NaN</td>
            </tr>
            <tr>
              <th>4</th>
              <td>peter</td>
              <td>33</td>
              <td>M</td>
              <td>engineer</td>
              <td>NaN</td>
            </tr>
            <tr>
              <th>5</th>
              <td>julie</td>
              <td>44</td>
              <td>F</td>
              <td>scientist</td>
              <td>171.0</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

Rename values


.. code-block:: default


    df.job = df.job.map({'student': 'etudiant', 'manager': 'manager',
                         'engineer': 'ingenieur', 'scientist': 'scientific'})









Dealing with outliers
---------------------


.. code-block:: default


    size = pd.Series(np.random.normal(loc=175, size=20, scale=10))
    # Corrupt the first 3 measures
    size[:3] += 500








Based on parametric statistics: use the mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume random variable follows the normal distribution
Exclude data outside 3 standard-deviations:
- Probability that a sample lies within 1 sd: 68.27%
- Probability that a sample lies within 3 sd: 99.73% (68.27 + 2 * 15.73)


.. code-block:: default


    size_outlr_mean = size.copy()
    size_outlr_mean[((size - size.mean()).abs() > 3 * size.std())] = size.mean()
    print(size_outlr_mean.mean())






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    248.48963819938044




Based on non-parametric statistics: use the median
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Median absolute deviation (MAD), based on the median, is a robust non-parametric statistics.
https://en.wikipedia.org/wiki/Median_absolute_deviation


.. code-block:: default


    mad = 1.4826 * np.median(np.abs(size - size.median()))
    size_outlr_mad = size.copy()

    size_outlr_mad[((size - size.median()).abs() > 3 * mad)] = size.median()
    print(size_outlr_mad.mean(), size_outlr_mad.median())






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    173.80000467192673 178.7023568870694




File I/O
--------

csv
~~~


.. code-block:: default


    import tempfile, os.path

    tmpdir = tempfile.gettempdir()
    csv_filename = os.path.join(tmpdir, "users.csv")
    users.to_csv(csv_filename, index=False)
    other = pd.read_csv(csv_filename)








Read csv from url
~~~~~~~~~~~~~~~~~


.. code-block:: default


    url = 'https://github.com/duchesnay/pystatsml/raw/master/datasets/salary_table.csv'
    salary = pd.read_csv(url)









Excel
~~~~~


.. code-block:: default


    xls_filename = os.path.join(tmpdir, "users.xlsx")
    users.to_excel(xls_filename, sheet_name='users', index=False)

    pd.read_excel(xls_filename, sheet_name='users')

    # Multiple sheets
    with pd.ExcelWriter(xls_filename) as writer:
        users.to_excel(writer, sheet_name='users', index=False)
        df.to_excel(writer, sheet_name='salary', index=False)

    pd.read_excel(xls_filename, sheet_name='users')
    pd.read_excel(xls_filename, sheet_name='salary')







.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>name</th>
              <th>age</th>
              <th>gender</th>
              <th>job</th>
              <th>height</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>0</th>
              <td>alice</td>
              <td>19</td>
              <td>F</td>
              <td>etudiant</td>
              <td>165.0</td>
            </tr>
            <tr>
              <th>1</th>
              <td>john</td>
              <td>26</td>
              <td>M</td>
              <td>etudiant</td>
              <td>180.0</td>
            </tr>
            <tr>
              <th>2</th>
              <td>eric</td>
              <td>22</td>
              <td>M</td>
              <td>etudiant</td>
              <td>175.0</td>
            </tr>
            <tr>
              <th>3</th>
              <td>paul</td>
              <td>58</td>
              <td>F</td>
              <td>manager</td>
              <td>NaN</td>
            </tr>
            <tr>
              <th>4</th>
              <td>peter</td>
              <td>33</td>
              <td>M</td>
              <td>ingenieur</td>
              <td>NaN</td>
            </tr>
            <tr>
              <th>5</th>
              <td>julie</td>
              <td>44</td>
              <td>F</td>
              <td>scientific</td>
              <td>171.0</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

SQL (SQLite)
~~~~~~~~~~~~


.. code-block:: default


    import pandas as pd
    import sqlite3

    db_filename = os.path.join(tmpdir, "users.db")








Connect


.. code-block:: default


    conn = sqlite3.connect(db_filename)








Creating tables with pandas


.. code-block:: default


    url = 'https://github.com/duchesnay/pystatsml/raw/master/datasets/salary_table.csv'
    salary = pd.read_csv(url)

    salary.to_sql("salary", conn, if_exists="replace")








Push modifications


.. code-block:: default


    cur = conn.cursor()
    values = (100, 14000, 5,  'Bachelor', 'N')
    cur.execute("insert into salary values (?, ?, ?, ?, ?)", values)
    conn.commit()









Reading results into a pandas DataFrame


.. code-block:: default


    salary_sql = pd.read_sql_query("select * from salary;", conn)
    print(salary_sql.head())

    pd.read_sql_query("select * from salary;", conn).tail()
    pd.read_sql_query('select * from salary where salary>25000;', conn)
    pd.read_sql_query('select * from salary where experience=16;', conn)
    pd.read_sql_query('select * from salary where education="Master";', conn)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

       index  salary  experience education management
    0      0   13876           1  Bachelor          Y
    1      1   11608           1      Ph.D          N
    2      2   18701           1      Ph.D          Y
    3      3   11283           1    Master          N
    4      4   11767           1      Ph.D          N


.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>index</th>
              <th>salary</th>
              <th>experience</th>
              <th>education</th>
              <th>management</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>0</th>
              <td>3</td>
              <td>11283</td>
              <td>1</td>
              <td>Master</td>
              <td>N</td>
            </tr>
            <tr>
              <th>1</th>
              <td>5</td>
              <td>20872</td>
              <td>2</td>
              <td>Master</td>
              <td>Y</td>
            </tr>
            <tr>
              <th>2</th>
              <td>6</td>
              <td>11772</td>
              <td>2</td>
              <td>Master</td>
              <td>N</td>
            </tr>
            <tr>
              <th>3</th>
              <td>9</td>
              <td>12313</td>
              <td>3</td>
              <td>Master</td>
              <td>N</td>
            </tr>
            <tr>
              <th>4</th>
              <td>11</td>
              <td>21371</td>
              <td>3</td>
              <td>Master</td>
              <td>Y</td>
            </tr>
            <tr>
              <th>5</th>
              <td>16</td>
              <td>12884</td>
              <td>4</td>
              <td>Master</td>
              <td>N</td>
            </tr>
            <tr>
              <th>6</th>
              <td>17</td>
              <td>13245</td>
              <td>5</td>
              <td>Master</td>
              <td>N</td>
            </tr>
            <tr>
              <th>7</th>
              <td>22</td>
              <td>13839</td>
              <td>6</td>
              <td>Master</td>
              <td>N</td>
            </tr>
            <tr>
              <th>8</th>
              <td>23</td>
              <td>22884</td>
              <td>6</td>
              <td>Master</td>
              <td>Y</td>
            </tr>
            <tr>
              <th>9</th>
              <td>25</td>
              <td>14803</td>
              <td>8</td>
              <td>Master</td>
              <td>N</td>
            </tr>
            <tr>
              <th>10</th>
              <td>30</td>
              <td>15942</td>
              <td>10</td>
              <td>Master</td>
              <td>N</td>
            </tr>
            <tr>
              <th>11</th>
              <td>32</td>
              <td>23780</td>
              <td>10</td>
              <td>Master</td>
              <td>Y</td>
            </tr>
            <tr>
              <th>12</th>
              <td>33</td>
              <td>25410</td>
              <td>11</td>
              <td>Master</td>
              <td>Y</td>
            </tr>
            <tr>
              <th>13</th>
              <td>35</td>
              <td>16882</td>
              <td>12</td>
              <td>Master</td>
              <td>N</td>
            </tr>
            <tr>
              <th>14</th>
              <td>38</td>
              <td>26330</td>
              <td>13</td>
              <td>Master</td>
              <td>Y</td>
            </tr>
            <tr>
              <th>15</th>
              <td>39</td>
              <td>17949</td>
              <td>14</td>
              <td>Master</td>
              <td>N</td>
            </tr>
            <tr>
              <th>16</th>
              <td>41</td>
              <td>27837</td>
              <td>16</td>
              <td>Master</td>
              <td>Y</td>
            </tr>
            <tr>
              <th>17</th>
              <td>42</td>
              <td>18838</td>
              <td>16</td>
              <td>Master</td>
              <td>N</td>
            </tr>
            <tr>
              <th>18</th>
              <td>44</td>
              <td>19207</td>
              <td>17</td>
              <td>Master</td>
              <td>N</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

Exercises
---------

Data Frame
~~~~~~~~~~

1. Read the iris dataset at 'https://github.com/neurospin/pystatsml/tree/master/datasets/iris.csv'

2. Print column names

3. Get numerical columns

4. For each species compute the mean of numerical columns and store it in  a ``stats`` table like:

::

          species  sepal_length  sepal_width  petal_length  petal_width
    0      setosa         5.006        3.428         1.462        0.246
    1  versicolor         5.936        2.770         4.260        1.326
    2   virginica         6.588        2.974         5.552        2.026


Missing data
~~~~~~~~~~~~

Add some missing data to the previous table ``users``:


.. code-block:: default


    df = users.copy()
    df.loc[[0, 2], "age"] = None
    df.loc[[1, 3], "gender"] = None








1. Write a function ``fillmissing_with_mean(df)`` that fill all missing
value of numerical column with the mean of the current columns.

2. Save the original users and "imputed" frame in a single excel file
"users.xlsx" with 2 sheets: original, imputed.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  1.794 seconds)


.. _sphx_glr_download_auto_gallery_scipy_pandas.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: scipy_pandas.py <scipy_pandas.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: scipy_pandas.ipynb <scipy_pandas.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
