import numpy as np
import matplotlib.pyplot as plt

I = np.eye(3)

g = 9.81  #Gravitational acceleration (m/s^2)
p = 1.225 #Air density at 20C (kg/m^3)
m = 0.484 #Mass of plane (kg)

# Inertia (kg*m^2)
Jx  = 0.003922
Jy  = 0.015940
Jz  = 0.019340
Jxz = 0.000441
J = np.array([[Jx, 0, Jxz],
              [0, Jy, 0],
              [Jxz, 0, Jz]])

# Main Wing
b = 86.4/100 #wing span (m)
cr = 26.0/100 #root chord (m)
ct = 15.2/100 #tip chord (m)
cm = (ct + cr)/2 #mean wing chord (m)
S = b*cm #planform area of wing (m^2)
Ra = b**2/S #wing aspect ratio (dimensionless)
Rt = ct/cr #wing taper ratio (dimensionless)
r_wing1 = np.array([0.0, (b/6)*(1+2*Rt)/(1+Rt), 0.0]) #vector from CoM to wing center-of-pressure (m)
r_wing2 = np.array([0.0, -(b/6)*(1+2*Rt)/(1+Rt), 0.0])
e_ail = 0.45 #flap effectiveness (dimensionless)

# Elevator
b_ele = 18.2/100 #elevator span (m)
cr_ele = 15.2/100 #elevator root chord (m)
ct_ele = 14/100 #elevator tip chord (m)
cm_ele = (ct_ele + cr_ele)/2 #mean elevator chord (m)
S_ele = b_ele*cm_ele #planform area of elevator (m^2)
Ra_ele = b_ele**2/S_ele #elevator aspect ratio (dimensionless)
r_ele = np.array([-45.0/100.0, 0.0, 0.0]) #vector from CoM to elevator center-of-pressure (m)
e_ele = 0.8 #flap effectiveness (dimensionless)

# Rudder
b_rud = 21.6/100 #rudder span (m)
cr_rud = 20.4/100 #rudder root chord (m)
ct_rud = 12.9/100 #rudder tip chord (m)
cm_rud = (ct_rud + cr_rud)/2 #mean rudder chord (m)
S_rud = b_rud*cm_rud #planform area of rudder (m^2)
Ra_rud = b_rud**2/S_rud #rudder aspect ratio (dimensionless)
r_rud = [-48.0/100.0, 0, -3.0/100.0] #vector from CoM to rudder center-of-pressure (m)
e_rud = 0.7 #flap effectiveness (dimensionless)

# Lift and drag polynomial coefficients from wind tunnel data
Clcoef = np.array([38.779513049043175, 0.0, 19.266141214863080, 0.0, -13.127972418509980, 0.0, 3.634063117174400, 0.0])
Cdcoef = np.array([3.607550808703421, 0.0, -4.489225907857385, 0.0, 3.480420330498847, 0.0, 0.063691497636087])

def hat(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def L(q):
    return np.array([[q[0], -q[1], -q[2], -q[3]],
                     [q[1],  q[0], -q[3],  q[2]],
                     [q[2],  q[3],  q[0], -q[1]],
                     [q[3], -q[2],  q[1],  q[0]]])

def R(q):
    return np.array([[q[0], -q[1], -q[2], -q[3]],
                     [q[1],  q[0],  q[3], -q[2]],
                     [q[2], -q[3],  q[0],  q[1]],
                     [q[3],  q[2], -q[1],  q[0]]])

H = np.vstack((np.zeros((1,3)), np.eye(3)))

def Cl_wing(a):
    Cl = np.dot(Clcoef, np.power(a, np.arange(len(Clcoef)-1, -1, -1)))
    return Cl

def Cd_wing(a):
    Cd = np.dot(Cdcoef, np.power(a, np.arange(len(Cdcoef)-1, -1, -1)))
    return Cd

def Cl_ele(a):
    Cl = (np.pi/2)*Ra_ele*a
    return Cl

def Cd_ele(a):
    Cd = (np.pi/4)*Ra_ele*a*a
    return Cd

def Cl_rud(B):
    Cl = (np.pi/2)*Ra_rud*B
    return Cl

def Cd_rud(B):
    Cd = (np.pi/4)*Ra_rud*B*B
    return Cd

def aoa(v):
    at_a = np.arctan2(v[2], v[0])
    return at_a

def ss(v):
    sideslip = np.arctan2(v[1], v[0])
    return sideslip

def aero_forces(v, omega, u):
    v_wing1 = v + hat(omega) @ r_wing1
    alpha_wing1 = aoa(v_wing1)
    alpha_eff_wing1 = alpha_wing1 - e_ail * u[0]
    L_wing1 = Cl_wing(alpha_eff_wing1) * 0.5 * p * np.linalg.norm(v_wing1)**2 * S
    D_wing1 = Cd_wing(alpha_eff_wing1) * 0.5 * p * np.linalg.norm(v_wing1)**2 * S
    F_wing1 = arotate(alpha_wing1, L_wing1, D_wing1)
    tau_wing1 = np.cross(r_wing1, F_wing1)

    v_wing2 = v + hat(omega) @ r_wing2
    alpha_wing2 = aoa(v_wing2)
    alpha_eff_wing2 = alpha_wing2 + e_ail * u[0]
    L_wing2 = Cl_wing(alpha_eff_wing2) * 0.5 * p * np.linalg.norm(v_wing2)**2 * S
    D_wing2 = Cd_wing(alpha_eff_wing2) * 0.5 * p * np.linalg.norm(v_wing2)**2 * S
    F_wing2 = arotate(alpha_wing2, L_wing2, D_wing2)
    tau_wing2 = np.cross(r_wing2, F_wing2)

    v_ele = v + hat(omega) @ r_ele
    alpha_ele = aoa(v_ele)
    alpha_eff_ele = alpha_ele - e_ele * u[1]
    L_ele = Cl_ele(alpha_eff_ele) * 0.5 * p * np.linalg.norm(v_ele)**2 * S_ele
    D_ele = Cd_ele(alpha_eff_ele) * 0.5 * p * np.linalg.norm(v_ele)**2 * S_ele
    F_ele = arotate(alpha_ele, L_ele, D_ele)
    tau_ele = np.cross(r_ele, F_ele)

    v_rud = v + hat(omega) @ r_rud
    Beta_rud = ss(v_rud)
    Beta_eff_rud = Beta_rud - e_rud * u[2]
    L_rud = Cl_rud(Beta_eff_rud) * 0.5 * p * np.linalg.norm(v_rud)**2 * S_rud
    D_rud = Cd_rud(Beta_eff_rud) * 0.5 * p * np.linalg.norm(v_rud)**2 * S_rud
    F_rud = Brotate(Beta_rud, L_rud, D_rud)
    tau_rud = np.cross(r_rud, F_rud)

    F = F_wing1 + F_wing2 + F_ele + F_rud
    tau = tau_wing1 + tau_wing2 + tau_ele + tau_rud

    return F, tau


def arotate(a, L, D):
    # Rotate by angle of attack from wind frame into body frame
    F = np.array([-D, 0, -L])
    Q = np.array([
        [np.cos(a), 0, -np.sin(a)],
        [0, 1, 0],
        [np.sin(a), 0, np.cos(a)]
    ])
    F_rot = Q @ F
    return F_rot

def Brotate(beta, L, D):
    # Rotate by sideslip angle from wind frame into body frame
    F = np.array([-D, -L, 0])
    Q = np.array([[np.cos(beta), -np.sin(beta), 0],
                  [np.sin(beta), np.cos(beta), 0],
                  [0, 0, 1]])
    F_rot = np.dot(Q, F)
    # print(F_rot)
    return F_rot

def glider_dynamics_q(x, u):
    # Unpack state vector
    r = x[0:3] # N frame
    q = x[3:7] # B to N
    v = x[7:10] # B frame
    w = x[10:13] # B frame
    F, tau = aero_forces(v, w, u)
    Q = H.T @ L(q) @ R(q).T @ H
    r_dot = Q @ v
    q_dot = 0.5 * L(q) @ H @ w
    v_dot = F / m - np.cross(w, v) - Q.T @ np.array([0, 0, g])
    w_dot = np.linalg.inv(J) @ (tau - np.cross(w, J @ w))
    return np.concatenate((r_dot, q_dot, v_dot, w_dot))

def normalized_rk4_step(xk, uk, h):
    q = xk[3:7] / np.linalg.norm(xk[3:7])
    xk[3:7] = q
    f1 = glider_dynamics_q(xk, uk)
    f2 = glider_dynamics_q(xk + 0.5 * h * f1, uk)
    f3 = glider_dynamics_q(xk + 0.5 * h * f2, uk)
    f4 = glider_dynamics_q(xk + h * f3, uk)
    xn = xk + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    return xn


Tf = 5.0
h = 1.0/50
tsamp = np.arange(0, Tf+h, h)
N = len(tsamp)

x0 = np.array([-10.0, 0.0, 10.0, 0.0, 0.9891199724086334, 0.0, -0.1471111150876924, 20.0,
               0.0, 0.5678625279339364, 0.0, 0.0, 0.0])

u0 = np.array([0.0, 0.27090666435590915, 0.0])

x0_loop = x0
u0_loop = u0

xhist_mid = np.zeros((13,N))
xhist_mid[:,0] = x0_loop

xhist_rk4 = np.zeros((13,N))
xhist_rk4[:,0] = x0_loop

for k in range(N-1):
    xhist_rk4[:,k+1] = normalized_rk4_step(xhist_rk4[:,k], u0_loop, h) # rk4 here

pos_3d = xhist_rk4[0:3, :]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(pos_3d[0, :], pos_3d[1, :], pos_3d[2, :])
plt.show()

