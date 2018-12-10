from PIL import Image
import numpy as np

# 高斯高通滤波器
def gass_high_H(m,n,rh,rl,c,D0):
    P = 2*m
    Q = 2*n
    img_homo = np.zeros((P, Q),dtype=np.float)
    a = D0**2
    r = rh-rl
    print(a,r)
    for u in range(P):
        for v in range(Q):
            tmp = (u-m)**2 + (v-n)**2
            img_homo[u][v] = r * (1-np.exp((-c)*(tmp/a))) + rl
    return img_homo


def getDistance(x, y, center_x, center_y):
    return np.sqrt((x-center_x)**2+(y-center_y)**2)

def Butterworth(cD):
    result = np.zeros((cD.shape[0], cD.shape[1]))
    center_x = cD.shape[0]/2
    center_y = cD.shape[1]/2
    for i in range(cD.shape[0]):
        for j in range(cD.shape[1]):
            if i == center_x and j == center_y:
                result[i][j] = 0
                continue
            temp = 0.95/getDistance(i, j, center_x, center_y)
            temp = temp**4
            gg = 1/(1+temp)
            result[i][j] = gg
    return result


# 归一到[0,L-1]
def normalize(image_in):
    m,n = image_in.shape 
    Dmax = image_in.max()
    Dmin = image_in.min()
    Dlen = Dmax - Dmin
    image_out = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            image_out[i][j] = np.uint8(255 * (image_in[i][j]- Dmin) / Dlen)
    return image_out

# 乘以(-1)的(x+y)次方转换中心
def Center(image_in, P, Q):
    m,n = image_in.shape
    image_out = np.zeros((P, Q),dtype = np.float64)
    for i in range(min(m,P)):
        for j in range(min(n,Q)):
            image_out[i][j] = np.double(image_in[i][j]) * (-1)**(i+j)
    return image_out

# 同态滤波函数
def HomoFilter(image_in, rh, rl, c, D0):
    m,n = image_in.size
    # 将图像转换成矩阵
    img = np.array(image_in)
    # 求导
    img = np.log(np.float64(img) + 1)
    # 转换中心
    img = Center(img, 2*m, 2*n)
    # 傅里叶变换
    img = np.fft.fft2(img)
    # 滤波操作
    img_r = img * gass_high_H(m,n,rh,rl,c,D0)
    img_i = img - img_r
    # 反傅里叶变换
    img = np.fft.ifft2(img)
    # 取实部
    img = np.real(img)
    # 变换回来
    img = Center(img,m,n)
    # 求指数回来
    image_out = np.exp(img) - 1
    # 归一化
    image_out = normalize(image_out)
    image_out = np.uint8(image_out)
    # 转换回图片
    image_out = Image.fromarray(image_out,'L')
    return image_out


def showImg(img):
    # 反傅里叶变换
    img = np.fft.ifft2(img)
    # 取实部
    img = np.real(img)
    # 变换回来
    img = Center(img, m, n)
    # 求指数回来
    image_out = np.exp(img) - 1
    # 归一化
    image_out = normalize(image_out)
    image_out = np.uint8(image_out)
    # 转换回图片
    image_out = Image.fromarray(image_out, 'L')
    return image_out


image_in = im_2
image_in
m, n = image_in.size
# 将图像转换成矩阵
img_refer = np.array(image_in)
# 求导
img_refer = np.log(np.float64(img_refer) + 1)
# 转换中心
img_refer = Center(img_refer, 2*m, 2*n)
# 傅里叶变换
img_refer = np.fft.fft2(img_refer)
# 滤波操作
# img_r = img_refer * gass_high_H(100,100,2,0.25,1,30)
img_r_2 = img_refer * Butterworth(img_refer)
img_i_2 = img_refer - img_r_2
