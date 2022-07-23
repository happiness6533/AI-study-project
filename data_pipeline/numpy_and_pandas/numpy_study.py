# numpy: 행렬 계산

# 1. 넘파이 배열
import numpy as np

# 행렬
a = np.array([[1, 2, 3],
              [1, 5, 9],
              [3, 5, 7]])
a = np.zeros((3, 3))
a = np.ones((3, 3))
a = np.full((3, 3), 4)
a = np.eye(3)  # 3x3 크기의 단위행렬 생성
a = np.random.random((2, 2))
a = np.arange(10, 30, 5)  # 10 이상 30 미만 간격 5
print(a)
print(a.shape)  # (3, 3)
print(a.dtype)  # dtype('int32')

# 벡터
a = np.array([1, 2, 3])
print(a)
print(a.shape)  # (,3)
print(a.dtype)  # dtype('int32')

# 연산: + - matmul dot T
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
C = np.array([[1, 2],
              [3, 4]])
D = np.array([[10, 20],
              [30, 10]])

print(np.matmul(C, D))  # 행렬 곱
print(np.dot(a, b))  # 벡터 내적
print(a.T)
print(C * D)  # element wise 곱
print(b < 10)
print(a ** 2)
print(C.sum())
print(C.sum(axis=0))
print(C.sum(axis=1))
print(C.max(axis=1))

# 인덱싱, 슬라이싱
a = np.array([1, 2, 3, 4, 5, 6, 7])
print(a[3])
print(a[-1])
print(a[2:5])
print(a[2:])
print(a[:4])

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(a[1, 2])  # 6
print(a[:, 1])  # 1열의 모든 원소 # array([2, 5, 8])
print(a[2, :])  # 마지막 행 # array([7, 8, 9])
print(a[-1])  # 마지막 행 # array([7, 8, 9])

# 형태 변환
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
print(a.reshape(2, 6))
print(a.reshape(12, ))  # 벡터
print(a.reshape(12, 1))  # 행렬
print(a.reshape(3, -1))  # 자동 재정렬

# 브로드캐스팅
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(a + 10)
