import numpy as np
import matplotlib.pyplot as plt

# input design parameters
_l_ = 0.051  # width of projectile (.051 full scale, .0381 prototype)
_Bf_ = .857  # permanent magnetic field (.8327 full scale, .857 prototype)
_R_ = .49  # resistance of system (.49 measured)
_C_ = .068*4  # total capacitance
_V_ = 60  # initial voltage
_m_ = .05  # mass of projectile
_X_ = .330  # length of barrel 12in = .305, 13 = .330, 18in = .457, 30in = .762

_R1_ = .01  # rail radius
_R2_ = _R1_ + _l_  # region 2
_L_ = 0.000002  # inductance of system
_mp_ = 2. * 10 ** (-7)  # permeability of vacuum divided by 2 pi

dt = 1. * 10 ** -6  # time step
N = 10 ** 5
param = [0]

# initialize time dependent variables
a = 0  # acceleration
x = 0  # position
Bl = 0  # induced mag field
F = 0  # force
L = 0  # inductance
# define lists for plotted time dependant variables
t = [0]  # time
v = [0]  # velocity
I = [0]  # current
Q = [_C_ * _V_]  # charge


# define functions
def func_Bl(_mp_, _R2_, _R1_, I, _l_):  # induced magnetic field
    return _mp_ * np.log(_R2_ / _R1_) * I / _l_


def func_F(I, _l_, _Bf_, Bl):  # force
    return I * _l_ * (_Bf_ + Bl)


def func_L(_mp_, _R2_, _R1_, x, _L_):  # inductance
    return (_mp_ * np.log(_R2_ / _R1_) * x) + _L_


def func_dIdt(L, Q, _C_, I, _R_, _Bf_, Bl, _l_, v):  # change in current
    return (1 / L) * ((Q / _C_) - (I * _R_) - ((_Bf_ + Bl) * _l_ * v))


for i in range(N):

    # calculate functions  (only need to call time dependant values in the functions)
    Bl = func_Bl(_mp_, _R2_, _R1_, I[-1], _l_)
    F = func_F(I[-1], _l_, _Bf_, Bl)
    a = F / _m_
    v.append(v[-1] + a * dt)
    x = x + v[-1] * dt
    L = func_L(_mp_, _R2_, _R1_, x, _L_)
    dIdt = func_dIdt(L, Q[-1], _C_, I[-1], _R_, _Bf_, Bl, _l_, v[-1])
    Q.append(Q[-1] - I[-1] * dt)
    I.append(I[-1] + dIdt * dt)

    #param.append(Bl)

    # iterate
    t.append(t[-1] + dt)

    # enforce boundary conditions
    if x >= _X_:
        break

Eff = (v[-1]*v[-1]*_m_ + (Q[-1]*Q[-1]/_C_))/(_C_*_V_*_V_)

# output final values
print("final t =", t[-1])
print("final v =", v[-1])
print("eff =", Eff)
print("final Q =", Q[-1])
print("avg I =", sum(I)/len(I))
print("max I =", max(I))

# output plots
fig, axs = plt.subplots(2)
axs[0].set_title("Velocity")
axs[0].plot(t, v)
axs[1].set_title("Current")
axs[1].plot(t, I)
plt.show()
