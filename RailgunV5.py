import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt

# This program simulates a capacitor-powered rail gun for a single set of design parameters.

# input design parameters
_l_ = 0.051  # [m] width of projectile (.051 full scale, .0381 prototype)
_m_ = .0211  # [kg] mass of projectile
_Bf_ = .854  # [T] permanent magnetic field (.8538 full scale, .857 prototype)
_X_ = .330  # [m] length of barrel 12in = .305, 13in = .330, 18in = .457, 30in = .762
_R1_ = .00127  # [m] rail radius (.1 in = .00254)
_C_ = 7 * .068  # [F] total capacitance (.068 F per capacitor)
_V_ = 50  # [V] initial voltage
_L_ = 2 * 10 ** (-6)  # [H] inductance of system (0.00000133)
_R_ = 0.01  # [ohms] resistance of system (.283 measured via .85V at 3A) (.05 predicted)
_fricoef_ = .1  # friction coefficient of greased copper
_rho_ = 0 #0.00053  # resistance per meter of rails
_Vb_ = 0 # breakdown voltage of small gap between projectile & rails

# universal constants
_mp_ = 2. * 10 ** (-7)  # permeability of vacuum divided by 2 pi (2x10^-7)
_g_ = 9.81  # gravity

# secondary calculations
_Lc_ = _mp_ * np.log((_l_ + _R1_) / _R1_)  # Inductance Constant

# simulation parameters
t_start = 0
t_stop = 1
t_steps = 10000
t_vec = np.linspace(t_start, t_stop, t_steps)

# initialize time dependent variables
x = 0.0  # projectile position
v = 0.0  # projectile velocity
Q = _C_ * _V_  # charge in capacitors
I = 0.0  # current


def dvdt_func(Q, I, _l_, _Bf_, _m_, _Lc_, _fricoef_, _g_):  # acceleration
    #if (1 / _m_) * ((I * _l_ * _Bf_) + (2 * _Lc_ * I * I)) >= (_fricoef_*_g_) and (Q/_C_) >= _Vb_:
    return (1 / _m_) * ((I * _l_ * _Bf_) + (2 * _Lc_ * I * I)) - (_fricoef_*_g_)
    #else:
    #    return 0


def dIdt_func(Q, _C_, I, _R_, _Bf_, _l_, v, x, _L_, _rho_):  # change in current
    return (1 / ((2 * _Lc_ * x) + _L_)) * ((Q / _C_) - (I * (_R_ + _rho_*x)) - (_Bf_ * _l_ * v))

# calculate derivatives of time dependant variables as vector
def railgun_func(t, y, _l_, _m_, _Bf_, _X_, _R1_, _L_, _Lc_, _R_, _rho_, _fricoef_, _g_):
    v = y[0]
    x = y[1]
    I = y[2]
    Q = y[3]

    dvdt = dvdt_func(Q, I, _l_, _Bf_, _m_, _Lc_, _fricoef_, _g_)
    dxdt = v
    dIdt = dIdt_func(Q, _C_, I, _R_, _Bf_, _l_, v, x, _L_, _rho_)
    dQdt = -I

    return [dvdt, dxdt, dIdt, dQdt]

# break integration when projectile reaches end of barrel
def check(t, y, _l_, _m_, _Bf_, _X_, _R1_, _L_, _Lc_, _R_, _rho_, _fricoef_, _g_):
    return y[1] - _X_
check.terminal = True


sol = scipy.integrate.solve_ivp(railgun_func, (t_start, t_stop), [v, x, I, Q], t_eval=t_vec, events=check, args=(_l_, _m_, _Bf_, _X_, _R1_, _L_, _Lc_, _R_, _rho_, _fricoef_, _g_), max_step = .000001)

v_list = sol.y[0]
x_list = sol.y[1]
I_list = sol.y[2]
Q_list = sol.y[3]
t_list = sol.t

# Energy Calculations
P0 = .5 * _C_ * _V_**2  # Initial Potential Energy
Pf = (.5 * Q_list[-1]**2) / _C_  # Final Potential Energy
K = .5 * _m_ * v_list[-1]**2  # Kinetic Energy of Projectile
I2_list = I_list[:]**2
U = (sum(I2_list)/(len(I_list))) * _R_ * t_list[-1]  # Thermal Energy lost to Resistance
W = .5 * ((2 * _Lc_ * x_list[-1]) + _L_) * I_list[-1]**2  # Energy stored in Magnetic field

# Curve Fit
def expon(x,a,b,c,d):
    return a*x*np.exp(-b*x)+c*x*np.exp(-d*x)

F_opt, F_cov = scipy.optimize.curve_fit(expon, t_list, (np.gradient(v_list)*_m_), p0=[80,1820,14.4,430])
a_opt = F_opt[0]
b_opt = F_opt[1]
c_opt = F_opt[2]
d_opt = F_opt[3]

print(F_opt)

# output final values
print(sol.message)
print("\nFinal Velocity =", v_list[-1])
print("Final Position =", x_list[-1])
print("Final Voltage =", Q_list[-1]/_C_)
print("Final Time =", t_list[-1])
print("avg current =", sum(I_list)/len(I_list))
print("max current =", max(I_list))
#print("max dIdt =", (I_list[1]-I_list[0])/(t_list[1]-t_list[0]))
print("\n")

print("ENERGY:")
print("Total Initial Energy = ", P0)
print("Total Final Energy = ", Pf + K + U + W)
print("Final Energy in Capacitors = ", Pf)
print("Kinetic Energy = ", K)
print("Energy Lost to Resistance = ", U)
print("Energy Stored in Magnetic Field = ", W)


# output plots
fig, axs = plt.subplots(3, dpi=200)
plt.xlabel('Time [s]')
axs[0].set_title("Velocity [m/s]")
axs[0].plot(t_list, v_list, '-k')
axs[1].set_title("Current [A]")
axs[1].plot(t_list, I_list, '-b')
axs[2].set_title("Force [N]")
axs[2].plot(t_list, (np.gradient(v_list)*_m_), '-b')
axs[2].plot(t_list, expon(t_list,a_opt,b_opt,c_opt,d_opt), '--k')
plt.show()
