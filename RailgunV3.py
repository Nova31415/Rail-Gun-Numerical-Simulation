import numpy as np
import matplotlib.pyplot as plt

# This program simulates a capacitor-powered rail gun.

# input design parameters
_l_ = 0.0521  # [m] width of projectile (.051 full scale, .0381 prototype)
_Bf_ = .854  # [T] permanent magnetic field (.8538 full scale, .857 prototype)
_R_ = .01  # [ohms] resistance of system (.283 measured via .85V at 3A) (.05 predicted)
_C_ = 4 * .068  # [F] total capacitance (.068 F/capacitor)
_V_ = 60  # [V] initial voltage
_m_ = .0179  # [kg] mass of projectile
_X_ = .330  # [m] length of barrel 12in = .305, 13in = .330, 18in = .457, 30in = .762
_R1_ = .00127  # [m] rail radius (.1 in = .00254)

_R2_ = _R1_ + _l_  # [m] region 2
_L_ = 0.00000133  # [H] inductance of system (0.00000133)
_mp_ = 2. * 10 ** (-7)  # permeability of vacuum divided by 2 pi (2x10^-7)
_Lc_ = _mp_ * np.log((_l_ + _R1_) / _R1_)  # Inductance Constant

dt = 1. * 10 ** -6  # [s] time step
t_final = 1
param = [0]

# initialize time dependent variables
a = 0.0  # acceleration
x = [0.0]  # position
Bl = 0.0  # induced mag field
F = 0.0  # force
L = 0.0  # inductance
# define lists for plotted time dependant variables
t = [0.0]  # time
v = [0.0]  # velocity
I = [0.0]  # current
Q = [_C_ * _V_]  # charge

def L_func(_Lc_, x): # self inductance
    return (2 * _Lc_ * x) + _L_

def a_func(I, _l_, _Bf_, _m_, _Lc_):  # acceleration
    return (1/_m_) * ((I * _l_ * _Bf_) + (_Lc_ * I * I))

def dIdt_func(Q, _C_, I, _R_, _Bf_, _l_, v, L):  # change in current
    return (1 / L) * ((Q / _C_) - (I * _R_) - (_Bf_ * _l_ * v))


while x[-1] < _X_:

    # calculate functions
    L = L_func(_Lc_, x[-1])
    dIdt = dIdt_func(Q[-1], _C_, I[-1], _R_, _Bf_, _l_, v[-1], L)
    I.append(I[-1] + dIdt * dt)
    Q.append(Q[-1] - I[-1] * dt)
    a = a_func(I[-1], _l_, _Bf_, _m_, _Lc_)
    v.append(v[-1] + a * dt)
    x.append(x[-1] + v[-1] * dt)

    # track any desired parameters
    param.append((L * dIdt) + (_Bf_*_l_*v[-1]))

    # iterate
    t.append(t[-1] + dt)

    # safety break
    if t[-1] > t_final:
        print("ERROR: TIME LIMIT REACHED")
        break

# Energy Calculations
P0 = .5 * _C_ * _V_**2  # Initial Potential Energy
Pf = (.5 * Q[-1]**2) / _C_  # Final Potential Energy
K = .5 * _m_ * v[-1]**2  # Kinetic Energy of Projectile
U = (sum(I)/len(I))**2 * _R_ * t[-1]  # Thermal Energy lost to Resistance
W = .5 * L * I[-1]**2


# output final values
print("final time =", t[-1])
print("final velocity =", v[-1])
#print("Final Energy Ratio = ", (.5 * _m_ * v[-1]**2 + (.5 * Q[-1]**2) / _C_ ) / (.5 * _C_ * _V_**2))
print("Initial Potential Energy = ", P0)
print("Final Potential Energy = ", Pf)
print("Kinetic Energy = ", K)
print("Energy Lost to Resistance = ", U)
print("Energy Stored in Magnetic Field = ", W)
print("Final Energy = ", Pf + K + U + W)
print("final Voltage =", Q[-1]/_C_)
print("avg current =", sum(I)/len(I))
print("max current =", max(I))

# output plots
fig, axs = plt.subplots(3, dpi=200)
plt.xlabel('Time [s]')
axs[0].set_title("Velocity [m/s]")
axs[0].plot(t, v, '-k')
axs[1].set_title("Current [A]")
axs[1].plot(t, I, '-b')
axs[2].set_title("Back EMF [V]")
axs[2].plot(t, param, '-r')
plt.show()
