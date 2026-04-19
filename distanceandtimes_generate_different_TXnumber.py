import random
import torch
import numpy as np
import pandas as pd
import time
from math import sqrt

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 分子接收函数
# def molecules_received(rx_x, rx_y, rx_z, xx, yy, zz, rx_radius):
#     distance = torch.sqrt((rx_x - xx)**2 + (rx_y - yy)**2 + (rx_z - zz)**2)
#     result = torch.lt(distance, rx_radius)
#     return result.all().item()

# 判断Tx坐标不要与Rx重叠
def molecules_received2(rx_x, rx_y, rx_z, xx, yy, zz, rx_radius):
    distance = torch.sqrt((rx_x - xx) ** 2 + (rx_y - yy) ** 2 + (rx_z - zz) ** 2)
    result = torch.lt(distance, rx_radius)
    return result.all().item()


x_bound = [x * 1e-6 for x in [-10, 10.1]]
y_bound = [y * 1e-6 for y in [-10, 10.1]]
z_bound = [z * 1e-6 for z in [-10, 10.1]]
resolution = 1 * 1e-6
x = np.arange(x_bound[0], x_bound[1], resolution)
y = np.arange(y_bound[0], y_bound[1], resolution)
z = np.arange(z_bound[0], z_bound[1], resolution)
X, Y, Z = np.meshgrid(x, y, z)
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
numberofemission=100000
# RX的坐标（0，0）
rx_x = torch.tensor([0.0] * numberofemission, dtype=torch.float, device=device) * 1e-6
rx_y = torch.tensor([0.0] * numberofemission, dtype=torch.float, device=device) * 1e-6
rx_z = torch.tensor([0.0] * numberofemission, dtype=torch.float, device=device) * 1e-6
rx_radius = torch.tensor(3, dtype=torch.float, device=device) * 1e-6

Txnumber=3
# 实验参数
delta_t = 1e-2
total_time = 2
T = 2

Diffision = []
for i in range(Txnumber):
    Diffision.append(1e-10)#正常1e-10
DD = 8e-14#正常1e-12
# 定义数组大小
shape = (120000, int((total_time / delta_t) * Txnumber+ Txnumber*2))

# 使用 NumPy 创建数组
data_m = np.zeros(shape, dtype=float)

# 将 NumPy 数组转换为 PyTorch 张量
data_m_torch = torch.tensor(data_m)
if torch.cuda.is_available():
    data_m_torch = data_m_torch.cuda()


class Molecule:
    def __init__(self):
        self.new_molecules_x = 0
        self.new_molecules_y = 1
        self.new_molecules_z = 2




start_time = time.time()

times = 0
flag = []
constant_term = []

for i in range(Txnumber):
    flag.append(False)
 #用来判断分子发射的时间布尔值
# 布朗运动常数(扩散系数不同)
for i in range(Txnumber):  # 因为这里只用一次3循环，所以用append添加完就可以，不需要用for循环进行内部index下的数字变换
    constant_term.append(torch.sqrt(torch.tensor(2.0 * Diffision[i] * delta_t, device=device)))
for i in range(Txnumber):
    # 初始化 tx_x, tx_y, tx_z 为长度为 3 的零数组
    tx_x = np.zeros(Txnumber)
    tx_y = np.zeros(Txnumber)
    tx_z = np.zeros(Txnumber)
    if times >=10000 :
        print("已经跑完全部10000")
        break
    else:
        print("开始跑")
    for txx, txy, txz in points:

        #Rx移动部分
        rx_x = torch.tensor([0.0] * numberofemission, dtype=torch.float, device=device) * 1e-6
        rx_y = torch.tensor([0.0] * numberofemission, dtype=torch.float, device=device) * 1e-6
        rx_z = torch.tensor([0.0] * numberofemission, dtype=torch.float, device=device) * 1e-6
        # 用来判断TX初始位置是否在RX的吸收半径内，在里面的话就舍弃这个TX
        if times >= 10000:
            break
        if not molecules_received2(rx_x, rx_y, rx_z, txx, txy, txz, rx_radius):
            # print(tx_x, tx_y, tx_z)
            for te in range(0, 100, 10):  # 每隔10个Δt，初始的TX开始发送分子（用来为模型判断不同发射时间）
                if times>=10000:
                    break
                emissiontime=[]
                for i in range(Txnumber):
                    emissiontime.append(te + random.randint(-5, 5))#随机时间
                    # emissiontime.append(te)#固定时间
                    if emissiontime[i] < 0: emissiontime[i] = te + random.randint(0, 5)

                print(f'times:{times}')
                for i in range(Txnumber):  # 用来分tx1、tx2、tx3的相关系数,这里是数据每一行形成的开始代码位置
                    tx_x[i] = txx
                    tx_y[i] = txy
                    tx_z[i] = txz
                    flag[i] = False#为每一行的flag都进行一个初始化，否则第一行发射完分子flag变成True后就无法再发射分子了
                    # if i == 0:  # 代表tx1
                    for t in range(0, int(T / delta_t)):#从0开始是因为TX就算没有发射分子也是要在0时刻就要开始移动了
                        if flag[i] == False and t==emissiontime[i]:#flag用来表示分子在te时刻发射分子后，就不会再发射分子了
                            molecules_x = torch.full((numberofemission,), tx_x[i].item(), device=device)  # 分子的发射位置就是TX的位置
                            molecules_y = torch.full((numberofemission,), tx_y[i].item(), device=device)
                            molecules_z = torch.full((numberofemission,), tx_z[i].item(), device=device)
                            flag[i] = True
                        if flag[i] == True: #flag=True时，代表分子已经发射了RX需要对分子的吸收进行一些判断
                            nx = torch.normal(mean=0.0, std=1.0, size=(numberofemission,), device=device)
                            ny = torch.normal(mean=0.0, std=1.0, size=(numberofemission,), device=device)
                            nz = torch.normal(mean=0.0, std=1.0, size=(numberofemission,), device=device)
                            # 分子在每个Δt内，通过布朗运动进行的位移
                            molecules_x += nx * constant_term[i]
                            molecules_y += ny * constant_term[i]
                            molecules_z += nz * constant_term[i]

                            distances = torch.sqrt(
                                (rx_x - molecules_x) ** 2 + (rx_y - molecules_y) ** 2 + (rx_z - molecules_z) ** 2)
                            # 判断每个分子是否被吸收
                            received = torch.lt(distances, rx_radius)
                            received_indices = torch.nonzero(received)  # 获取被吸收的分子的索引
                            received_count = received_indices.size(0)
                            data_m_torch[times, t+200*i] += received_count
                            # 将被吸收的分子的位置设置为无穷远
                            molecules_x[received_indices] = float('inf')
                            molecules_y[received_indices] = float('inf')
                            molecules_z[received_indices] = float('inf')
                        tensor_x = torch.tensor(tx_x[i])
                        tensor_y = torch.tensor(tx_y[i])
                        tensor_z = torch.tensor(tx_z[i])

                        delta_x = torch.normal(mean=0.0, std=1.0, size=(1,), device=device)
                        delta_y = torch.normal(mean=0.0, std=1.0, size=(1,), device=device)
                        delta_z = torch.normal(mean=0.0, std=1.0, size=(1,), device=device)

                        tx_x[i] = tensor_x + torch.sqrt(torch.tensor(2.0 * DD * delta_t, device=device)) * delta_x
                        tx_y[i] = tensor_y + torch.sqrt(torch.tensor(2.0 * DD * delta_t, device=device)) * delta_y
                        tx_z[i] = tensor_z + torch.sqrt(torch.tensor(2.0 * DD * delta_t, device=device)) * delta_z
                        # # RX变成动态的代码
                        # delta_x_rx = torch.normal(mean=0.0, std=1.0, size=(1,), device=device)
                        # delta_y_rx = torch.normal(mean=0.0, std=1.0, size=(1,), device=device)
                        # delta_z_rx = torch.normal(mean=0.0, std=1.0, size=(1,), device=device)
                        # rx_x = rx_x + torch.sqrt(torch.tensor(2.0 * DD * delta_t, device=device)) * delta_x_rx
                        # rx_y = rx_y + torch.sqrt(torch.tensor(2.0 * DD * delta_t, device=device)) * delta_y_rx
                        # rx_z = rx_z + torch.sqrt(torch.tensor(2.0 * DD * delta_t, device=device)) * delta_z_rx
                        if t==emissiontime[i]:
                            # # #Rx移动用到的式子
                            # data_m_torch[times, int((total_time / delta_t)) * Txnumber+i] = torch.sqrt((tensor_x-rx_x[0]) ** 2 + (tensor_y-rx_y[0]) ** 2 + (tensor_z-rx_z[0]) ** 2)*1e6
                            #正常用到的式子
                            data_m_torch[times, int((total_time / delta_t)) * Txnumber + i] = torch.sqrt(tensor_x ** 2 + tensor_y** 2 + tensor_z ** 2) * 1e6
                        data_m_torch[times, int((total_time / delta_t) * Txnumber + Txnumber+i)] = emissiontime[i]
                times = times + 1
df = pd.DataFrame(data_m_torch[:times].cpu().numpy())
df.to_csv('3_by_1_times_movetx_random_te_transformer_numberofemission=100000_banjing=4.csv', index=False)
print("Data saved to 'capture_times_data15.csv'")

end_time = time.time()
duration = end_time - start_time
print(f"Total execution time: {duration} seconds")
