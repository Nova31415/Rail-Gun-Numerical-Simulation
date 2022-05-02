import numpy as np
import matplotlib.pyplot as plt

# TO USE: Input 'Default' design parameters in the first block. To track how a specific paramter effects final velocity, input it into lines with 'CHANGE' on the right

# read in experimental results for comparison
# Cex,Vex,velex, err = np.genfromtxt("Velocity vs Voltage Data.txt", delimiter=",", skip_header=1, usecols=(0,1,6,7), unpack=True)

# define design parameters
_l_ = 0.0381  # width of projectile (.051 full scale, .0381 prototype)
_Bf_ = .857  # permanent magnetic field (.924 full scale, .6076 prototype)
_R_ = .49  # resistance of system (.49 measured)
_C_ = .068*12  # total capacitance
_V_ = 34  # initial voltage
_m_ = .0041  # mass of projectile
_X_ = .305  # length of barrel 12in = .305 18in = .457 30in = .762

# define constants
_R1_ = .00317  # rail radius
_L_ = 0.0002  # inductance of system
_mp_ = 2. * 10 ** (-7)  # permeability of vacuum divided by 2 pi
#_fricoef_ = .3  # friction coefficient of greased copper
#_rho_ = .01  # resistance per meter of rails
#_g_ = 9.81  # gravity

# simulation loop values
dt = 1. * 10 ** -6  # time step
N = 10 ** 8  # number of time steps

# simulation loop outputs
vf = []  # final velocity
tf = []  # final time
Qf = []  # remaining charge
If = []  # final current
Imax = []  # max current during firing
par = []  # parameter loop iteration


# define functions
def func_Bl(_R2_, I, _l_):  # induced magnetic field
    return _mp_ * np.log(_R2_ / _R1_) * I / _l_


def func_F(I, _l_, _Bf_, Bl):  # force
    return I * _l_ * (_Bf_ + Bl)


def func_L(_mp_, _R2_, _R1_, x, _L_):  # inductance
    return (_mp_ * np.log(_R2_ / _R1_) * x) + _L_


def func_dIdt(L, Q, _C_, I, _R_, _Bf_, Bl, _l_, v):  # change in current
    return (1 / L) * ((Q / _C_) - (I * _R_) - ((_Bf_ + Bl) * _l_ * v))


for _V_ in np.arange(23, 61, 1):  # for [parameter] in np.arange([min], [max], [step])                   CHANGE

    #  define starting value of second variable if neccessary
    #_X_ = .762

    # define parameter dependant variables
    #_m_ = (1.5 * (_l_ ** 2) * _h_) * _density_  # for mass of projectile being proportional to barrel dimensions, ignoring for now
    _R2_ = _R1_ + _l_  # rail center to edge of other rail

    # define time dependent variables
    a = 0  # acceleration
    x = 0  # position
    #R = _R_  # for accounting for changing resistance
    Bl = 0  # induced mag field
    F = 0  # force
    L = 0  # inductance

    # define lists for plotted time dependant variables
    t = [0]  # time
    v = [0]  # velocity
    I = [0]  # current
    Q = [_C_ * _V_]  # charge

    for i in range(N):  # simulation loop

        # calculate functions
        Bl = func_Bl(_R2_, I[-1], _l_)
        F = func_F(I[-1], _l_, _Bf_, Bl)
        #if F <= _fricoef_ * _m_ * _g_:     # a = 0 unless force > static friction, ignoring for now
        #    a = 0
        #else:
        a = F / _m_
        v.append(v[-1] + a * dt)
        x = x + v[-1] * dt
        #R = _R_ + _rho_ * x # update resistance as circuit expands, ignoring for now
        L = func_L(_mp_, _R2_, _R1_, x, _L_)
        dIdt = func_dIdt(L, Q[-1], _C_, I[-1], _R_, _Bf_, Bl, _l_, v[-1])
        Q.append(Q[-1] - I[-1] * dt)
        I.append(I[-1] + dIdt * dt)

        # iterate
        t.append(t[-1] + dt)

        # enforce barrel length
        if x >= _X_:
            break

    #log final values into meta lists
    vf.append(v[-1])
    #tf.append(t[-1]) for tracking time to reach end of barrel
    Qf.append(Q[-1])  # for tracking charge left in capacitors after firing
    #Imax.append(max(I)) for tracking max current reached during firing
    #_X_ = _X_ - .00457  # iterate second changing parameter
    par.append(_V_)  # track changing parameter                                                               CHANGE
    print(par[-1])  # track progress

# output plots on parameter axis
plt.plot(par, vf, label = "Theoretical")
#plt.scatter(Vex, velex, label = "Experimental")  # graph experimental data
#plt.errorbar(Vex, velex, yerr=err, fmt="o", label = "Experimental")
plt.title("Velocity As Function of Voltage")
plt.ylabel('Velocity [m/s]')
plt.xlabel('Voltage [V]')
plt.legend()
plt.show()
