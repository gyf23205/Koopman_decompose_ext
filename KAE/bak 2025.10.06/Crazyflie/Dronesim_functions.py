import numpy as np
import matplotlib.pyplot as plt


# ================================
# Utility
# ================================

import numpy as np

def hat(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def exp_so3(w, dt):
    th = np.linalg.norm(w * dt)
    if th < 1e-8:
        return np.eye(3)
    k = (w * dt) / th
    K = hat(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


class DroneSim:
    """
    12-state quadrotor simulator with full PID:
    - Velocity PID: P on velocity error, I on integrated error, D on measured acceleration
    - Attitude PID: P on attitude error, I on integrated error, D on measured angular velocity
    """

    def __init__(self, mass, J, dt,
                 init_pos_range, init_vel_range, init_omega_range,
                 pos_bounds, vel_bounds, omega_bounds,
                 Kp_v, Ki_v, Kd_v,
                 Kp_att, Ki_att, Kd_att,
                 windup_limit_v, windup_limit_att,
                 rng):

        self.m = mass
        self.J = J
        self.invJ = np.linalg.inv(J)
        self.dt = dt
        self.g = 9.81

        # random initial state
        self.p = rng.uniform(low=init_pos_range[0], high=init_pos_range[1], size=3)
        self.v = rng.uniform(low=init_vel_range[0], high=init_vel_range[1], size=3)
        self.R = np.eye(3)
        self.omega = rng.uniform(low=init_omega_range[0], high=init_omega_range[1], size=3)

        # bounds
        self.pos_bounds = np.array(pos_bounds)
        self.vel_bounds = np.array(vel_bounds)
        self.omega_bounds = np.array(omega_bounds)

        # PID gains
        self.Kp_v, self.Ki_v, self.Kd_v = np.array(Kp_v), np.array(Ki_v), np.array(Kd_v)
        self.Kp_att, self.Ki_att, self.Kd_att = np.array(Kp_att), np.array(Ki_att), np.array(Kd_att)

        # integrator states
        self.int_ev = np.zeros(3)
        self.int_eR = np.zeros(3)

        # windup limits
        self.windup_limit_v = windup_limit_v
        self.windup_limit_att = windup_limit_att

        # measured acceleration for D-term
        self.accel = np.zeros(3)

    # -------------------------------------------------

    def velocity_controller(self, v_des):
        """Outer loop: desired velocity -> desired force/attitude"""
        ev = v_des - self.v

        # integral term with anti-windup
        if self.windup_limit_v >= 0:
            self.int_ev += ev * self.dt
            self.int_ev = np.clip(self.int_ev, -self.windup_limit_v, self.windup_limit_v)
        else:
            self.int_ev[:] = 0.0

        # PID: D on measured accel
        a_des = (self.Kp_v * ev +
                 self.Ki_v * self.int_ev -
                 self.Kd_v * self.accel)

        f_des = self.m * (a_des + np.array([0, 0, self.g]))
        f_mag = np.linalg.norm(f_des) + 1e-9
        z_b_des = f_des / f_mag

        # construct desired attitude R_des
        x_c = np.array([1, 0, 0])
        if abs(np.dot(z_b_des, x_c)) > 0.9:
            x_c = np.array([0, 1, 0])
        y_b_des = np.cross(z_b_des, x_c); y_b_des /= (np.linalg.norm(y_b_des) + 1e-9)
        x_b_des = np.cross(y_b_des, z_b_des)
        R_des = np.column_stack([x_b_des, y_b_des, z_b_des])

        return f_mag, R_des

    # -------------------------------------------------

    def attitude_controller(self, R_des):
        """Inner loop: desired attitude -> torque"""
        R_err = 0.5 * (R_des.T @ self.R - self.R.T @ R_des)
        e_R = np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])

        # integral term with anti-windup
        if self.windup_limit_att >= 0:
            self.int_eR += e_R * self.dt
            self.int_eR = np.clip(self.int_eR, -self.windup_limit_att, self.windup_limit_att)
        else:
            self.int_eR[:] = 0.0

        tau = -1* (self.Kp_att * e_R +
               self.Ki_att * self.int_eR +
               self.Kd_att * self.omega)
        return tau

    # -------------------------------------------------

    def step(self, v_des):
        """Simulate one time step given desired velocity"""
        # controllers
        T, R_des = self.velocity_controller(v_des)
        tau = self.attitude_controller(R_des)

        # translational dynamics
        f_thrust = T * (self.R @ np.array([0, 0, 1]))
        f_total = f_thrust - self.m * np.array([0, 0, self.g])
        a = f_total / self.m
        self.v += a * self.dt
        self.p += self.v * self.dt
        self.accel = a.copy()  # store accel for next D-term

        # rotational dynamics
        omega_dot = self.invJ @ (tau - np.cross(self.omega, self.J @ self.omega))
        self.omega += omega_dot * self.dt
        self.R = self.R @ exp_so3(self.omega, self.dt)

        # enforce bounds
        self.p = np.clip(self.p, -self.pos_bounds, self.pos_bounds)
        self.v = np.clip(self.v, -self.vel_bounds, self.vel_bounds)
        self.omega = np.clip(self.omega, -self.omega_bounds, self.omega_bounds)

        # ✅ return all four states
        return self.p.copy(), self.v.copy(), self.R.copy(), self.omega.copy()



# ================================
# Infinity Pattern
# ================================

def lemniscate_velocity(x, a, k_p, k_t, v_max):
    X, Y = x[0], x[1]
    f = (X**2 + Y**2)**2 - 2*(a**2)*(X**2 - Y**2)
    gradf = np.array([4*X*(X**2+Y**2) - 4*a**2*X,
                      4*Y*(X**2+Y**2) + 4*a**2*Y])
    gradV = f * gradf
    T = np.array([-gradf[1], gradf[0]])
    T /= np.linalg.norm(T) + 1e-8
    v_des = -k_p * gradV + k_t * T
    v_norm = np.linalg.norm(v_des)
    if v_norm > v_max:
        v_des = v_des / v_norm * v_max
    return np.array([v_des[0], v_des[1], 0.0])


# ================================
# Simulation
# ================================

def run_simulation(steps, dt, sim_params, pattern_params):
    sim = DroneSim(**sim_params)
    traj, v_log, v_des_log = [], [], []
    for _ in range(steps):
        v_des = lemniscate_velocity(sim.p, **pattern_params)
        p, v, R, w = sim.step(v_des)
        traj.append(p)
        v_log.append(v)
        v_des_log.append(v_des)
    return np.array(traj), np.array(v_log), np.array(v_des_log)


# ================================
# Visualization
# ================================

def plot_trajectory(traj):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=2, color="blue")
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c="green", label="Start")
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c="red", label="End")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(); ax.set_title("Drone Infinity-Shape Trajectory")
    plt.show()

def plot_velocity(v_log, v_des_log, dt):
    t = np.arange(len(v_log)) * dt
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["vx", "vy", "vz"]
    for i in range(3):
        axs[i].plot(t, v_log[:, i], label=f"{labels[i]} actual")
        axs[i].plot(t, v_des_log[:, i], "--", label=f"{labels[i]} desired")
        axs[i].legend(); axs[i].set_ylabel("m/s")
    axs[-1].set_xlabel("Time [s]")
    plt.suptitle("Velocity Tracking (PID + PID)"); plt.show()


