import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# This program plots the muzzle velocity of a capacitor-driven railgun calculated for a range of two design parameters
# To change which parameters to vary, change the corresponding variable name and range in the lines marked 'CHANGE'
fig = plt.figure(figsize=[9,5], dpi =200)
# read in experimental results
#Vex = [20,20,20, 24,24,24, 28,28,28, 32,32,32, 36,36,36, 40,40,40, 44,44,44, 48,48,48, 52,52,52, 56,56,56, 60,60,60]
#velex = [13.8,15.36,16.2, 20.4,21.6,20.4, 26.4,27.48,28.32, 32,33.6,34.08, 36.64,38.08,38.56, 41.44,42.96,42.48, 43.2,47.04,48, 50.16,51.12,51.36, 51.84,53.28,54, 55.92,56.16,56.16, 59.52,58.8,56.88]
#plt.plot(Vex, velex, '.b',  label = "Experimental",)

# input design parameters
_l_ = 0.051  # [m] width of projectile (.051 full scale, .0381 prototype)
_m_ = .0211  # [kg] mass of projectile
_Bf_ = .854  # [T] permanent magnetic field (.8538 full scale, .857 prototype)
_X_ = .330  # [m] length of barrel 12in = .305, 13in = .330, 18in = .457, 30in = .762
_R1_ = .00127  # [m] rail radius (.1 in = .00254)
_C_ = 7 * .068  # [F] total capacitance (.068 F/capacitor)
_V_ = 60  # [V] initial voltage
_L_ = 50 * 10 ** (-6)  # [uH] inductance of system
_R_ = 0.011  # [ohms] resistance of system

# universal constants
_mp_ = 2. * 10 ** (-7)  # permeability of vacuum divided by 2 pi (2x10^-7)

# secondary calculations
#_Lc_ = _mp_ * np.log((_l_ + _R1_) / _R1_)  # Inductance Constant

# simulation parameters
t_start = 0
t_stop = 1


# define functions
def dvdt_func(I, _l_, _Bf_, _m_, _Lc_, x):  # acceleration
    if x < _X_:
        return (1 / _m_) * ((I * _l_ * _Bf_) + (_Lc_ * I * I))
    else:
        return 0

def dIdt_func(Q, _C_, I, _R_, _Bf_, _l_, v, x, _L_):  # change in current
    return (1 / ((2 * _Lc_ * x) + _L_)) * ((Q / _C_) - (I * _R_) - (_Bf_ * _l_ * v))

def railgun_func(t, z, _l_, _m_, _Bf_, _X_, _R1_, _L_, _Lc_, _R_):
    v = z[0]
    x = z[1]
    I = z[2]
    Q = z[3]

    dvdt = dvdt_func(I, _l_, _Bf_, _m_, _Lc_, x)
    dxdt = v
    dIdt = dIdt_func(Q, _C_, I, _R_, _Bf_, _l_, v, x, _L_)
    dQdt = -I

    return [dvdt, dxdt, dIdt, dQdt]

def check(t, z, _l_, _m_, _Bf_, _X_, _R1_, _L_, _Lc_, _R_):
    return z[1] - _X_
check.terminal = True


for _L_ in np.logspace(-7,-3,5, base=10):   # for [parameter 1] in numpy.linspace([min], [max], [num steps])                   CHANGE

    # meta lists
    vf = []
    par = []
    v_list = []

    for _V_ in np.linspace(20, 60, 41):  # for [parameter 2] in numpy.linspace([min], [max], [num steps])                   CHANGE

        # initialize time/parameter dependent variables
        x = 0.0  # projectile position
        v = 0.0  # projectile velocity
        Q = _C_ * _V_  # charge in capacitors
        I = 0.0  # current
        _Lc_ = _mp_ * np.log((_l_ + _R1_) / _R1_)  # Inductance Constant

        # ode solver
        sol = scipy.integrate.solve_ivp(railgun_func, (t_start, t_stop), [v, x, I, Q], dense_output=True, events=check, args=(_l_, _m_, _Bf_, _X_, _R1_, _L_, _Lc_, _R_), max_step = .0001)

        # unpack values
        v_list = sol.y[0]
        t_list = sol.t

        if sol.status == 0:
            print("TIME OUT ERROR")

        #log final values into meta lists
        vf.append(v_list[-1])
        par.append(_V_)  # track changing parameter 2                                                              CHANGE
        print(par[-1])  # track progress

    plt.plot(par, vf, '-', label = _L_,)

#opt_param_index = np.argmax(vf)
#opt_param = par[opt_param_index]

#print("Optimal Parameter Value = ", opt_param)


#plt.plot(Vex_rej, velex_rej, '.r',  label = "Experimental (Rejected)")
#plt.errorbar(Vex, velex, yerr=err, fmt="o", label = "Experimental")
plt.title("Velocity As Function of Voltage For Different Inductance Values")
plt.ylabel('Velocity [m/s]')
plt.xlabel('Voltage [V]')
plt.legend()
plt.show()
