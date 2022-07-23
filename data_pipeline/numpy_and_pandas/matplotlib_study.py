# matplotlib: 그림 그리기

import numpy as np
import matplotlib.pyplot as plt

# 그래프 그리기
x = [1, 3, 5, 7, 9]
y = [100, 200, 300, 400, 500]
plt.plot(x, y)
plt.show()

x = np.linspace(- np.pi, np.pi, 128)  # 연속적인 값을 갖는 배열
y = np.cos(x)  # x 리스트에 대한 cos값 계산
plt.plot(x)
plt.show()

plt.plot(y)
plt.show()

# 판다스 데이터 시각화
import pandas as pd

data_frame = pd.read_csv('./data_in/datafile.csv')  # 데이터를 읽어온다.
data_frame.plot()
data_sum = data_frame.cumsum()  # 데이터를 누적값으로 바꿔준다.
data_sum.plot()
