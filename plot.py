# -*- coding: utf-8 -*-
"""
@author: Biao
"""
import matplotlib.pyplot as plt
import os
PRICE = [0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.12,0.14,0.14,0.12,0.14,0.14,0.14,0.14,0.12,0.12,0.12,0.12,0.12,0.12,0.06]
fig1 = plt.figure(1)
plt.plot(PRICE)
plt.ylabel('Price($/Wh)')
plt.xlabel('Time(hour)')
if not os.path.exists('image'):
    os.makedirs('image')
plt.savefig(os.path.join('image', '_'.join(['Price'])))

T_OUT = [23,23,23,23,23,24,24,26,27,28,30,31,32,34,33,32,32,30,30,29,27,24,23,23]
fig2 = plt.figure(2)
plt.plot(T_OUT)
plt.ylabel('Outdoor Temperature(℃)')
plt.xlabel('Time(hour)')
if not os.path.exists('image'):
    os.makedirs('image')
plt.savefig(os.path.join('image', '_'.join(['T_out'])))

[10200, 8700]
[1278, 932]

