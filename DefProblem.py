# -*- coding: utf-8 -*-
"""
@author: Biao
"""

import numpy as np

#TIME   0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23
PRICE = [0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.12,0.14,0.14,0.12,0.14,0.14,0.14,0.14,0.12,0.12,0.12,0.12,0.12,0.12,0.06]
T_OUT = [23,23,23,23,23,24,24,26,27,28,30,31,32,34,33,32,32,30,30,29,27,24,23,23]
T_IN_0 = 23
T_MIN = 23 # comfortable minimum temperature 
T_MAX = 25 # comfortable maximum temperature 
K = 200 # cost of dissatisfaction

def init_s():
    s = np.array([0,PRICE[0],T_OUT[0],T_IN_0])
    return s

def step_s_DDPG(a,s):
    T_in = ((s[2] - s[3]) - (a+500)/100) * 0.6 + s[3]
    s_ = np.array([s[0]+1, PRICE[int(s[0]+1)], T_OUT[int(s[0]+1)], T_in],dtype = np.float32)
    r = reward_DDPG(a,s)
    return s_, r

def reward_DDPG(a,s):
    if (s[3] > T_MAX):
        cost = s[1] * (a+500) + K * (s[3] - T_MAX)
    elif (s[3] < T_MIN):
        cost = s[1] * (a+500) + K * (T_MIN - s[3])
    else:
        cost = s[1] * (a+500)
    return - cost

def step_s(a,s):
    T_in = ((s[2] - s[3]) - a) * 0.6 + s[3]
    s_ = np.array([s[0]+1, PRICE[int(s[0]+1)], T_OUT[int(s[0]+1)], T_in])
    r = reward(a,s)
    return s_, r

def reward(a,s):
    if (s[3] > T_MAX):
        cost = s[1] * a * 100 + K * (s[3] - T_MAX)
    elif (s[3] < T_MIN):
        cost = s[1] * a * 100 + K * (T_MIN - s[3])
    else:
        cost = s[1] * a * 100
    return  -cost

if __name__ == '__main__':
    L_a = []
    for i in range(24):
        L_a.append((T_OUT[i] - 23) * 100 * PRICE[i])
    print(sum(L_a))