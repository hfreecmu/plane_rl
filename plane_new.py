import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import matplotlib.pyplot as plt

control_dim = 4
trim_controls = np.array([41.666, 106, 74.6519, 106])

class YakPlane():
    def __init__(self):
        self.g = 9.81 # Gravitational acceleration (m/s^2)
        self.rho = 1.2 # Air density at 20C (kg/m^3)
        self.m = 0.075 # Mass of plane (kg)

        self.Jx = 4.8944e-04 # Roll axis inertia (kg*m^2)
        self.Jy = 6.3778e-04 # Pitch axis inertia (kg*m^2)
        self.Jz = 7.9509e-04 # Yaw axis inertia (kg*m^2)

        self.J: np.ndarray = np.diag([self.Jx, self.Jy, self.Jz]) # Moment of inertia matrix
        self.Jinv: np.ndarray = np.diag([1/self.Jx, 1/self.Jy, 1/self.Jz]) # Inverse of moment of inertia matrix (assuming products of inertia are small)

        self.Jm = 0.007*(0.0075)**2 + 0.002*(0.14)**2/12 # Motor + prop inertia (kg*m^2)

        # All lifting surfaces are modeled as unsweapt tapered wings
        self.b = 45/100 # Wing span (m)
        self.l_in = 6/100 # Inboard wing length covered by propwash (m)
        self.cr = 13.5/100 # Root chord (m)
        self.ct = 8/100 # Tip chord (m)
        self.cm = (self.ct + self.cr)/2 # Mean wing chord (m)
        self.S = self.b*self.cm # Planform area of wing (m^2)
        self.S_in = 2*self.l_in*self.cr
        self.S_out = self.S - self.S_in
        # Ra = b**2/S # Wing aspect ratio (dimensionless)
        self.Rt = self.ct/self.cr # Wing taper ratio (dimensionless)
        self.r_ail = (self.b/6)*(1+2*self.Rt)/(1+self.Rt) # Aileron moment arm (m)

        self.ep_ail = 0.63 # Flap effectiveness (Phillips P.41)
        self.trim_ail = 106 # Control input for zero deflection
        self.g_ail = (15*np.pi/180)/100 # Maps control input to deflection angle

        self.b_elev = 16/100 # Elevator span (m)
        self.cr_elev = 6/100 # Elevator root chord (m)
        self.ct_elev = 4/100 # Elevator tip chord (m)
        self.cm_elev = (self.ct_elev + self.cr_elev)/2 # Mean elevator chord (m)
        self.S_elev = self.b_elev*self.cm_elev # Planform area of elevator (m^2)
        self.Ra_elev = self.b_elev**2/self.S_elev # Wing aspect ratio (dimensionless)
        self.r_elev = 22/100 # Elevator moment arm (m)

        self.ep_elev = 0.88 # Flap effectiveness (Phillips P.41)
        self.trim_elev = 106 # Control input for zero deflection
        self.g_elev = (20*np.pi/180)/100 #maps control input to deflection angle

        self.b_rud = 10.5/100 #rudder span (m)
        self.cr_rud = 7/100 #rudder root chord (m)
        self.ct_rud = 3.5/100 #rudder tip chord (m)
        self.cm_rud = (self.ct_rud + self.cr_rud)/2 #mean rudder chord (m)
        self.S_rud = self.b_rud*self.cm_rud #planform area of rudder (m^2)
        self.Ra_rud = self.b_rud**2/self.S_rud #wing aspect ratio (dimensionless)
        self.r_rud = 24/100 #rudder moment arm (m)
        self.z_rud = 2/100 #height of rudder center of pressure (m)

        self.ep_rud = 0.76 #flap effectiveness (Phillips P.41)
        self.trim_rud = 106 #control input for zero deflection
        self.g_rud = (35*np.pi/180)/100 #maps from control input to deflection angle

        self.trim_thr = 24 #control input for zero thrust (deadband)
        self.g_thr = 0.006763 #maps control input to Newtons of thrust
        self.g_mot = 3000*2*np.pi/60*7/255 #maps control input to motor rad/sec

        self.H = np.vstack((np.zeros((1,3)), np.eye(3)))

def dynamics(p, x, u):
    r = x[0:3]
    q = x[3:7]
    v = x[7:10]
    w = x[10:]

    #Q = Rotation.from_quat(q).as_matrix()
    Q = p.H.T @ L(q) @ R(q).T @ p.H

    # control input
    thr, ail, elev, rud = u

    # ------- Input Checks -------- #
    thr = np.clip(thr, 0, 255)
    ail = np.clip(ail, 0, 255)
    elev = np.clip(elev, 0, 255)
    rud = np.clip(rud, 0, 255)

    # ---------- Map Control Inputs to Angles ---------- #
    delta_ail = (ail-p.trim_ail)*p.g_ail
    delta_elev = (elev-p.trim_elev)*p.g_elev
    delta_rud = (rud-p.trim_rud)*p.g_rud

    # ---------- Aerodynamic Forces (body frame) ---------- #
    v_body = Q.T @ v  # body-frame velocity
    v_rout = v_body + np.cross(w, np.array([0, p.r_ail, 0]))
    v_lout = v_body + np.cross(w, np.array([0, -p.r_ail, 0]))
    v_rin = v_body + np.cross(w, np.array([0, p.l_in, 0])) + propwash(thr)
    v_lin = v_body + np.cross(w, np.array([0, -p.l_in, 0])) + propwash(thr)
    v_elev = v_body + np.cross(w, np.array([-p.r_elev, 0, 0])) + propwash(thr)
    v_rud = v_body + np.cross(w, np.array([-p.r_rud, 0, -p.z_rud])) + propwash(thr)

    # --- Outboard Wing Sections --- #
    a_rout = alpha(v_rout)
    a_lout = alpha(v_lout)
    a_eff_rout = a_rout + p.ep_ail*delta_ail  # effective angle of attack
    a_eff_lout = a_lout - p.ep_ail*delta_ail  # effective angle of attack

    F_rout = -p_dyn(p, v_rout)*0.5*p.S_out*np.array([Cd_wing(a_eff_rout), 0, Cl_wing(a_eff_rout)])
    F_lout = -p_dyn(p, v_lout)*0.5*p.S_out*np.array([Cd_wing(a_eff_lout), 0, Cl_wing(a_eff_lout)])

    F_rout = arotate(a_rout, F_rout)  # rotate to body frame
    F_lout = arotate(a_lout, F_lout)  # rotate to body frame

    # --- Inboard Wing Sections (Includes Propwash) --- #
    a_rin = alpha(v_rin)
    a_lin = alpha(v_lin)
    a_eff_rin = a_rin + p.ep_ail*delta_ail  # effective angle of attack
    a_eff_lin = a_lin - p.ep_ail*delta_ail  # effective angle of attack

    F_rin = -p_dyn(p, v_rin)*0.5*p.S_in*np.array([Cd_wing(a_eff_rin), 0, Cl_wing(a_eff_rin)])
    F_lin = -p_dyn(p, v_lin)*0.5*p.S_in*np.array([Cd_wing(a_eff_lin), 0, Cl_wing(a_eff_lin)])

    F_rin = arotate(a_rin, F_rin)  # rotate to body frame
    F_lin = arotate(a_lin, F_lin)  # rotate to body frame

    # --- Elevator --- #
    a_elev = alpha(v_elev)
    a_eff_elev = a_elev + p.ep_elev*delta_elev  # effective angle of attack

    F_elev = -p_dyn(p, v_elev)*p.S_elev*np.array([Cd_elev(p, a_eff_elev), 0, Cl_plate(a_eff_elev)])

    F_elev = arotate(a_elev, F_elev)  # rotate to body frame

    # --- Rudder --- #
    a_rud = beta(v_rud)
    a_eff_rud = a_rud - p.ep_rud*delta_rud  # effective angle of attack

    F_rud = -p_dyn(p, v_rud)*p.S_rud*np.array([Cd_rud(p, a_eff_rud), Cl_plate(a_eff_rud), 0])

    F_rud = brotate(a_rud, F_rud)  # rotate to body frame

    # --- Thrust --- #
    if thr > p.trim_thr:
        F_thr = np.array([(thr-p.trim_thr)*p.g_thr, 0, 0])
        w_mot = np.array([p.g_mot*thr, 0, 0])
    else: #deadband
        F_thr = np.zeros(3)
        w_mot = np.zeros(3)

    # ---------- Aerodynamic Torques (body frame) ---------- #
    T_rout = np.cross(np.array([0, p.r_ail, 0]), F_rout)
    T_lout = np.cross(np.array([0, -p.r_ail, 0]), F_lout)

    T_rin = np.cross(np.array([0, p.l_in, 0]), F_rin)
    T_lin = np.cross(np.array([0, -p.l_in, 0]), F_lin)

    T_elev = np.cross(np.array([-p.r_elev, 0, 0]), F_elev)

    T_rud = np.cross(np.array([-p.r_rud, 0, -p.z_rud]), F_rud)

    # ---------- Add Everything Together ---------- #
    # problems: F_lout, F_rin
    F_aero = F_rout + F_lout + F_rin + F_lin + F_elev + F_rud + F_thr
    F = Q @ F_aero - np.array([0, 0, p.m * p.g])

    T = T_rout + T_lout + T_rin + T_lin + T_elev + T_rud + np.cross((p.J @ w + p.Jm * w_mot), w)

    rdot = v
    qdot = 0.5 * L(q) @ p.H @ w
    vdot = F / p.m
    wdot = np.dot(p.Jinv, T)

    return np.concatenate((rdot, qdot, vdot, wdot))

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

# Angle of attack
def alpha(v):
    return np.arctan2(v[2], v[0])

# Sideslip angle
def beta(v):
    return np.arctan2(v[1], v[0])

# Rotate by angle of attack
def arotate(a, r):
    sa, ca = np.sin(a), np.cos(a)
    R = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
    return np.dot(R, r)

# Rotate by sideslip angle
def brotate(b, r):
    sb, cb = np.sin(b), np.cos(b)
    R = np.array([[cb, -sb, 0], [sb, cb, 0], [0, 0, 1]])
    return np.dot(R, r)

""" Propwash wind speed (body frame)
Fit from anemometer data taken at tail
No significant different between wing/tail measurements
"""

def propwash(thr):
    trim_thr = 24  # control input for zero thrust (deadband)
    if thr > trim_thr:
        v = np.array([5.568*thr**0.199 - 8.859, 0, 0])
    else:  # deadband
        v = np.zeros(3)
    return v

def p_dyn(p, v):
    pd = 0.5 * p.rho * np.dot(v, v)
    return pd

""" Lift coefficient (alpha in radians)
3rd order polynomial fit to glide-test data
Good to about ±20°
"""
def Cl_wing(a):
    a = max(-0.5*np.pi, min(a, 0.5*np.pi))
    cl = -27.52*a**3 - 0.6353*a**2 + 6.089*a
    return cl

""" Lift coefficient (alpha in radians)
Ideal flat plate model used for wing and rudder
"""
def Cl_plate(a):
    a = max(-0.5*np.pi, min(a, 0.5*np.pi))
    cl = 2*np.pi*a
    return cl

""" Drag coefficient (alpha in radians)
2nd order polynomial fit to glide-test data
Good to about ±20°
"""
def Cd_wing(a):
    a = max(-0.5*np.pi, min(a, 0.5*np.pi))
    cd = 2.08*a**2 + 0.0612
    return cd

""" Drag coefficient (alpha in radians)
Induced drag for a tapered finite wing
    From phillips P.55
"""
def Cd_elev(p, a):
    a = max(-0.5*np.pi, min(a, 0.5*np.pi))
    cd = (4*np.pi*a**2)/p.Ra_elev
    return cd

""" Drag coefficient (alpha in radians)
Induced drag for a tapered finite wing
From Phillips P.55
"""
def Cd_rud(p, a):
    a = max(-0.5*np.pi, min(a, 0.5*np.pi))
    cd = (4*np.pi*a**2)/p.Ra_rud
    return cd

def normalized_rk4_step(p, xk, uk, h):
    q = xk[3:7] / np.linalg.norm(xk[3:7])
    xk[3:7] = q
    f1 = dynamics(p, xk, uk)
    f2 = dynamics(p, xk + 0.5 * h * f1, uk)
    f3 = dynamics(p, xk + 0.5 * h * f2, uk)
    f4 = dynamics(p, xk + h * f3, uk)
    xn = xk + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

    return xn

#load
x_path = '/home/frc-ag-3/harry_ws/courses/grad_ai/final_project/trajectories/iLQR_loop/iLQR_Result.csv'
df_x = pd.read_csv(x_path, header=None)
x_ilqr = df_x.values

u_path = '/home/frc-ag-3/harry_ws/courses/grad_ai/final_project/trajectories/iLQR_loop/loop_iLQR_controls.csv'
df_u = pd.read_csv(u_path, header=None)
u_ilqr = df_u.values

Tf = 2.5
h = 1.0/80
tsamp = np.arange(0, Tf+h, h)
N = len(tsamp)

xhist_rk4 = np.zeros((13, N))
xhist_rk4[:,0] = x_ilqr[0, :]

p = YakPlane()
for k in range(N-1):
    xhist_rk4[:,k+1] = normalized_rk4_step(p, xhist_rk4[:,k], u_ilqr[k], h) # rk4 here

pos_3d = xhist_rk4[0:3, :]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(pos_3d[0, :], pos_3d[1, :], pos_3d[2, :])
ax.scatter(x_ilqr[:, 0], x_ilqr[:, 1], x_ilqr[:, 2])

breakpoint()

plt.show()