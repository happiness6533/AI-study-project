# 판다스: 데이터 => 표(시리즈, 데이터프레임, 판넬)

import numpy as np
import pandas as pd

# 1. 시리즈 생성
a = pd.Series([1, 3, 5, 7, 10])
print(a)
b = pd.Series(np.array(['a', 'b', 'c', 'd']))
print(b)
c = pd.Series(np.arange(10, 30, 5))
print(c)
a = pd.Series(['a', 'b', 'c'], index=[10, 20, 30])
print(a)
d = pd.Series({'a': 10, 'b': 20, 'c': 30})
print(d)

# 2. 데이터 프레임 생성
a = pd.DataFrame([[1, 3, 5],
                  [2, 4, 6]])  # 리스트를 이용한 생성
print(a)

b = pd.DataFrame({'Name': ['Cho', 'Kim', 'Lee'], 'Age': [28, 31, 38]})  # 딕셔너리를 이용한 생성
print(b)

c = pd.DataFrame([['apple', 7000],
                  ['banana', 5000],
                  ['orange', 4000]])  # 리스트의 중첩에 의한 생성
print(c)

a = pd.DataFrame([['apple', 7000], ['banana', 5000], ['orange', 4000]], columns=['name', 'price'])
print(a)

# 판다스 데이터 불러오기 및 쓰기
data_frame = pd.read_csv('./data_in/datafile.csv')
print(data_frame['A'])  # A열의 데이터만 확인
print(data_frame['A'][:10])  # A열의 데이터 중 앞의 10개만 확인
data_frame['D'] = data_frame['A'] + data_frame['B']  # A열과 B열을 더한 새로운 D열 생성
print(data_frame['D'])
data_frame.describe()
