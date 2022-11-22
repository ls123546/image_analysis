import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def ff(x):
    x[x > 255] = 255
    x[x < 0] = 0
    return np.array(x, dtype=np.uint8)
def linear(a, b, x):
    result = a * x + b
    return ff(result)
def fenduan_linear( a, b, c, d, x):
    x[(b <= x) & (x <= a)] = (x[(b <= x) & (x <= a)] ) -10
    x[(c <= x) & (x <= b)] = 1.2* (x[(c <= x) & (x <= b)] ) +20
    x[(d <= x) & (x <= c)] = 2* x[(d <= x) & (x <= c)]+15
    return ff(x)

def nonlinear(a, b, x):
    return ff(a * x ** b)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
image = np.array(Image.open('1.png'))
fig = plt.figure(figsize=(18, 12))
plt.subplot(221)
plt.title('原图', fontsize=15)
plt.imshow(image)
plt.subplot(222)
plt.title('线性', fontsize=15)
plt.imshow(linear(1.5, 10, image))
plt.subplot(223)
plt.title('非线性', fontsize=15)
plt.imshow(nonlinear(2, 1.5, image))
plt.subplot(224)
plt.title('分段线性', fontsize=15)
plt.imshow(fenduan_linear(230, 200, 150, 100, image))
plt.show()
