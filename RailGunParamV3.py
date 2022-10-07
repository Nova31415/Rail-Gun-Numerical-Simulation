import numpy as np
import matplotlib.pyplot as plt

# This program plots the muzzle velocity of a capacitor-driven railgun calculated for a range of design parameter values.
# To change which parameter to vary, change the corresponding variable name and range in the lines marked 'CHANGE'

# read in experimental results
#Cex,Vex,velex, err = np.genfromtxt("Velocity vs Voltage Data.txt", delimiter=",", skip_header=1, usecols=(0,1,6,7), unpack=True)
#Vex = [24, 24, 24, 30, 30, 30, 36, 36, 36, 42, 42, 48, 60, 60, 60, 60]
#velex = [12.57, 7.58, 9.88, 17.85, 19.77, 20.16, 26.28, 25.44, 28.68, 31.2, 34.176, 42.96, 73.92, 71.36, 69.76, 67.84]
# rejected data
#Vex_rej = [42, 42, 48, 48, 48, 54, 54, 54]
#velex_rej = [17.76, 11.04, 16.56, 22.68, 14.4, 26.88, 41.28, 26.16]



# input design parameters
_l_ = 0.0521  # [m] width of projectile (.051 full scale, .0381 prototype)
_Bf_ = .854  # [T] permanent magnetic field (.8538 full scale, .857 prototype)
_R_ = .01  # [ohms] resistance of system (.283 measured via .85V at 3A) (.05 predicted)
_C_ = 4 * .068  # [F] total capacitance (.068 F/capacitor)
_V_ = 60  # [V] initial voltage
_m_ = .0176  # [kg] mass of projectile
_X_ = .330  # [m] length of barrel 12in = .305, 13in = .330, 18in = .457, 30in = .762
_R1_ = .00127  # [m] rail radius (.1 in = .00254)

# input non-parameter constants
_R2_ = _R1_ + _l_  # [m] region 2
_L_ = 0.00000133  # [H] inductance of system (0.00000133)
_mp_ = 2. * 10 ** (-7)  # permeability of vacuum divided by 2 pi (2x10^-7)
_Lc_ = _mp_ * np.log((_l_ + _R1_) / _R1_)  # Inductance Constant

# time step variables
dt = 1. * 10 ** -6  # [s] time step
t_final = 1

# meta lists
vf = []
par = []

# define functions
def L_func(_Lc_, x): # self inductance
    return (2 * _Lc_ * x) + _L_

def a_func(I, _l_, _Bf_, _m_, _Lc_):  # acceleration
    return (1/_m_) * ((I * _l_ * _Bf_) + (_Lc_ * I * I))

def dIdt_func(Q, _C_, I, _R_, _Bf_, _l_, v, L):  # change in current
    return (1 / L) * ((Q / _C_) - (I * _R_) - (_Bf_ * _l_ * v))


for _V_ in np.arange(24, 60, 1):  # for [parameter] in numpy.arange([min], [max], [step])                   CHANGE

    # initialize time dependent variables
    a = 0  # acceleration
    Bl = 0  # induced mag field
    L = 0  # inductance

    # initialize time dependant lists
    x = [0]  # position
    t = [0]  # time
    v = [0]  # velocity
    I = [0]  # current
    Q = [_C_ * _V_]  # charge

    while x[-1] < _X_:

        # calculate functions
        L = L_func(_Lc_, x[-1])
        dIdt = dIdt_func(Q[-1], _C_, I[-1], _R_, _Bf_, _l_, v[-1], L)
        I.append(I[-1] + dIdt * dt)
        Q.append(Q[-1] - I[-1] * dt)
        a = a_func(I[-1], _l_, _Bf_, _m_, _Lc_)
        v.append(v[-1] + a * dt)
        x.append(x[-1] + v[-1] * dt)

        # iterate
        t.append(t[-1] + dt)
        # safety break
        if t[-1] > t_final:
            print("ERROR: TIME LIMIT REACHED")
            break

    #log final values into meta lists
    vf.append(v[-1])
    #tf.append(t[-1])
    #Qf.append(Q[-1])
    #Imax.append(max(I))
    #_X_ = _X_ - .00457  # iterate second changing parameter
    par.append(_V_)  # track changing parameter                                                               CHANGE
    print(par[-1])  # track progress

# output plots on parameter axis
fig = plt.figure(figsize=[9,5], dpi =200)
plt.plot(par, vf, '-k', label = "Theoretical",)
#plt.plot(Vex, velex, '.b',  label = "Experimental",)
#plt.plot(Vex_rej, velex_rej, '.r',  label = "Experimental (Rejected)")
#plt.errorbar(Vex, velex, yerr=err, fmt="o", label = "Experimental")
plt.title("Velocity As Function of Voltage")
plt.ylabel('Velocity [m/s]')
plt.xlabel('Initial Voltage [V]')
plt.legend()
plt.show()
