import numpy as np
from numpy.linalg import det, inv, svd


def rect_integral(time_steps, f, args):
    I = 0    
    for i in range(len(time_steps)-1):
        I += (time_steps[i+1] - time_steps[i]) * f((time_steps[i] + time_steps[i+1]) / 2, args)
    return I


def trap_integral(time_steps, f, args):
    I = 0    
    for i in range(len(time_steps)-1):
        I += (time_steps[i+1] - time_steps[i]) * (f(time_steps[i], args) + f(time_steps[i+1], args)) / 2
    return I


def find_ind(arr, el):
    arr = np.array(arr)
    i = np.abs(arr - el).argmin()
    return i


def runge_kutta(time_steps, y0, system, params):
    ys = [y0]
    for t in range(len(time_steps)-1):
        dt = time_steps[t+1]-time_steps[t]
        t0 = time_steps[t]
        t1 = time_steps[t+1]
        k1 = system(t0, y0, params)
        k2 = system(t0 + dt/2, y0 + dt / 2 * k1, params)
        k3 = system(t0 + dt/2, y0 + dt / 2 * k2, params)
        k4 = system(t1, y0 + dt * k3, params)
        y0  = y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y0)
    return np.array(ys)


def shooting(time_steps, y_approx, system, params, bc, bc_params, solver=runge_kutta):
    eps = 10**(-4)
    t_left = time_steps[len(time_steps)//2::-1]
    t_right = time_steps[len(time_steps)//2:]
    newton_steps = 0
    F = np.zeros(len(y_approx)*len(y_approx)).reshape(len(y_approx), len(y_approx))
    
    while(True):
        print(newton_steps)
        ys = np.concatenate((solver(t_left, y_approx, system, params)[::-1],
              solver(t_right, y_approx, system, params)[1:]))
        rs = bc(bc_params, ys)
        print(rs)
        print(np.sum(np.abs(rs)))
        if (np.abs(rs) < eps).all():
            break
        
        F = np.zeros(len(y_approx)*len(y_approx)).reshape(len(y_approx), len(y_approx))
        for i in range(len(y_approx)):
            yi_approx = y_approx.copy()
            yi_approx[i] += eps
            
            yis = np.concatenate((solver(t_left, yi_approx, system, params)[::-1],
                   solver(t_right, yi_approx, system, params)[1:]))
            rsi = bc(bc_params, yis)
            columni = (rsi - rs) / eps
            F[:, i] = columni
        newton_steps += 1
        if det(F) == 0:
            print("AA")
            return newton_steps, ys, det(F), svd(F)[1]
        y_approx =  y_approx - np.dot(inv(F), rs)
    ys = np.concatenate((solver(t_left, y_approx, system, params)[::-1],
                   solver(t_right, y_approx, system, params)[1:]))
    return newton_steps, ys, det(F), svd(F)[1]

def augmented_frechet_matrix(time_steps, ys_sol, system, params, cur_p, bc, bc_params, solver=runge_kutta):
    eps = 10**(-4)
    t_left = time_steps[len(time_steps)//2::-1]
    t_right = time_steps[len(time_steps)//2:]
    y_init = ys_sol[t_left[0]]
    ys = np.concatenate((solver(t_left, y_init, system, params)[::-1],
                         solver(t_right, y_init, system, params)[1:]))
    rs = bc(bc_params, ys)
    aug_F = np.zeros(len(y_init)*len(y_init) + len(y_init)).reshape(len(y_init), len(y_init)+1)
    for i in range(len(y_init)):
        yi_init = y_init.copy()
        yi_init[i] += eps
        
        yis = np.concatenate((solver(t_left, yi_init, system, params)[::-1],
               solver(t_right, yi_init, system, params)[1:]))
        rsi = bc(bc_params, yis)

        columni = (rsi - rs) / eps
        aug_F[:, i] = columni
    params[cur_p] += eps
    yps = np.concatenate((solver(t_left, y_init, system, params)[::-1],
               solver(t_right, y_init, system, params)[1:]))
    params[cur_p] -= eps
    rsp = bc(bc_params, yps)
    columnp = (rsp - rs) / eps
    aug_F[:, -1] = columnp
    return aug_F

def analyse_point(aug_F):
    eps = 10**(-4)
    dets = []
    for col in range(len(aug_F[0])):
        cols = [i for i in range(len(aug_F[0])) if i != col]
        dets.append(det(aug_F[:, cols]))
    if (np.abs(dets) < eps).all():
        return True, np.round(dets, 4)
    return False, np.round(dets, 4)


def continuation_parameter(Xs, Ys, x_cur, y_cur, x_next):
    if len(Ys) < 3:
        Ys.append(y_cur)
        Xs.append(x_cur)
    else:
        Ys[0], Ys[1], Ys[2] = Ys[1], Ys[2], y_cur
        Xs[0], Xs[1], Xs[2] = Xs[1], Xs[2], x_cur
    a0 = Ys[0]
    if len(Ys) == 1: return a0
    a1 = (Ys[1] - Ys[0])/(Xs[1]-Xs[0])
    if len(Ys) == 2: return a0 + a1*(x_next-Xs[0])
    yx1x2 = (Ys[2]-Ys[1])/(Xs[2]-Xs[1])
    a2 = (yx1x2 - a1)/(Xs[2]-Xs[0])
    y_next = a0 + a1*(x_next-Xs[0]) + a2*(x_next-Xs[0])*(x_next-Xs[1])
    return y_next