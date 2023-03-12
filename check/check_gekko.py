from gekko import GEKKO

m = GEKKO()  # create GEKKO model
x = m.Var()  # define new variable, default=0
y = m.Var()  # define new variable, default=0
m.Equations([3 * x + 2 * y == 1, x + 2 * y == 0])  # equations
m.solve(disp=False)  # solve
print(x.value, y.value)  # print solution

# ---------------------------------------------------------------------------
# from gekko import GEKKO
m = GEKKO()  # create GEKKO model
x = m.Var(value=0)  # define new variable, initial value=0
y = m.Var(value=1)  # define new variable, initial value=1
m.Equations([x + 2 * y == 0, x ** 2 + y ** 2 == 1])  # equations
m.solve(disp=False)  # solve
print([x.value[0], y.value[0]])  # print solution

# ---------------------------------------------------------------------------
# from gekko import GEKKO
m = GEKKO()
p = m.Param(1.2)
x = m.Array(m.Var, 3)
eq0 = x[1] == x[0] + p
eq1 = x[2] - 1 == x[1] + x[0]
m.Equation(x[2] == x[1] ** 2)
m.Equations([eq0, eq1])
m.solve()
for i in range(3):
    print('x[' + str(i) + ']=' + str(x[i].value))

# ---------------------------------------------------------------------------
# from gekko import GEKKO

# Initialize Model
m = GEKKO(remote=True)

#help(m)

#define parameter
eq = m.Param(value=40)

#initialize variables
x1,x2,x3,x4 = [m.Var() for i in range(4)]

#initial values
x1.value = 1
x2.value = 5
x3.value = 5
x4.value = 1

#lower bounds
x1.lower = 1
x2.lower = 1
x3.lower = 1
x4.lower = 1

#upper bounds
x1.upper = 5
x2.upper = 5
x3.upper = 5
x4.upper = 5

#Equations
m.Equation(x1*x2*x3*x4>=25)
m.Equation(x1**2+x2**2+x3**2+x4**2==eq)

#Objective
m.Obj(x1*x4*(x1+x2+x3)+x3)

#Set global options
m.options.IMODE = 3 #steady state optimization

#Solve simulation
m.solve() # solve on public server

#Results
print('')
print('Results')
print('x1: ' + str(x1.value))
print('x2: ' + str(x2.value))
print('x3: ' + str(x3.value))
print('x4: ' + str(x4.value))

# ---------------------------------------------------------------------------
# from gekko import GEKKO
m = GEKKO()           # create GEKKO model
y = m.Var(value=2)    # define new variable, initial value=2
m.Equation(y**2==1)   # define new equation
m.options.SOLVER=1    # change solver (1=APOPT,3=IPOPT)
m.solve(disp=False)   # solve locally (remote=False)
print('y: ' + str(y.value)) # print variable value


# ---------------------------------------------------------------------------
# from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

m = GEKKO()
m.time = np.linspace(0,20,41)

# Parameters
mass = 500
b = m.Param(value=50)
K = m.Param(value=0.8)

# Manipulated variable
p = m.MV(value=0, lb=0, ub=100)
p.STATUS = 1  # allow optimizer to change
p.DCOST = 0.1 # smooth out gas pedal movement
p.DMAX = 20   # slow down change of gas pedal

# Controlled Variable
v = m.CV(value=0)
v.STATUS = 1  # add the SP to the objective
m.options.CV_TYPE = 2 # squared error
v.SP = 40     # set point
v.TR_INIT = 1 # set point trajectory
v.TAU = 5     # time constant of trajectory

# Process model
m.Equation(mass*v.dt() == -v*b + K*b*p)

m.options.IMODE = 6 # control
m.solve(disp=False)

# get additional solution information
import json
with open(m.path+'//results.json') as f:
    results = json.load(f)

plt.figure()
plt.subplot(2,1,1)
plt.plot(m.time,p.value,'b-',label='MV Optimized')
plt.legend()
plt.ylabel('Input')
plt.subplot(2,1,2)
plt.plot(m.time,results['v1.tr'],'k-',label='Reference Trajectory')
plt.plot(m.time,v.value,'r--',label='CV Response')
plt.ylabel('Output')
plt.xlabel('Time')
plt.legend(loc='best')
plt.show()
