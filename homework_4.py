import cv2
import numpy as np
import matplotlib.pyplot as plt
def MinFilterGray(src, r=17):
    return cv2.erode(src, np.ones((2*r+1, 2*r+1)))
def guidedfilter(I, p, r, eps):
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I*p, -1, (r, r))
    cov_Ip = m_Ip-m_I*m_p
    m_II = cv2.boxFilter(I*I, -1, (r, r))
    var_I = m_II-m_I*m_I
    a = cov_Ip/(var_I+eps)
    b = m_p-a*m_I
    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a*I+m_b
def getV1(m, r, eps, w, maxV1):
    V1 = np.min(m, 2)
    V1 = guidedfilter(V1, MinFilterGray(V1, 7), r, eps)
    bins = 2000
    ht = np.histogram(V1, bins)
    d = np.cumsum(ht[0])/float(V1.size)
    for lmax in range(bins-1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1*w, maxV1)
    return V1, A
def deHaze(m, r=27, eps=0.001, w=0.9, maxV1=0.8):
    Y = np.zeros(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)
    for k in range(3):
        Y[:, :, k] = (m[:, :, k]-V1)/(1-V1/A)
    Y = np.clip(Y, 0, 1)
    Y = Y**(np.log(0.5)/np.log(Y.mean()))
    return Y
img_in = cv2.imread('2.png')
defog_gamma = deHaze(img_in / 255.0) * 255
img_in = cv2.cvtColor(img_in.astype(np.uint8), cv2.COLOR_BGR2RGB)
defog_gamma = cv2.cvtColor(defog_gamma.astype(np.uint8), cv2.COLOR_BGR2RGB)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(18, 12))
plt.subplot(121)
plt.title('原图', fontsize=15)
plt.imshow(img_in)
plt.subplot(122)
plt.title('去雾后', fontsize=15)
plt.imshow(defog_gamma)
plt.show()
