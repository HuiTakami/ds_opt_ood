import scipy as sp
from utils_ds.my_check_function import my_check_function
from math import *
from casadi import *


def optimize_P(Data):
    d = int(len(Data) / 2)
    x = Data[:d, :]
    xd = Data[d:, :]

    p0 = cov_initial_guess(x)
    print('the target function originally is ', my_check_function(Data, p0))
    print('the eigenvalue of initial guess is ', np.linalg.eigvals(vector_to_matrix(p0)))
    if d == 3:
        p = SX.sym('y', 6)
        lambda_1, lambda_2, lambda_3 = calculate_eigenvalue(p)
        g = vertcat(lambda_1 - 0.1, lambda_2 - 0.1, lambda_3 - 0.1, lambda_1 + lambda_2 + lambda_3 - 1)
    else:
        p = SX.sym('y', 3)
        lambda_1, lambda_2 = calculate_eigenvalue_2D(p)
        g = vertcat(lambda_1 - 0.1, lambda_2 - 0.1, lambda_1 + lambda_2 - 1)
    nlp = {'x': p, 'f': object_function(p, x, xd, 0.0001), 'g': g}
    S = nlpsol('S', 'ipopt', nlp)
    if len(x) == 3:
        r = S(x0=p0,
              lbg=[0.00, 0.00, 0.00, 0], ubg=[inf, inf, inf, 0])
    else:
        r = S(x0=p0,
              lbg=[0.00, 0.00, 0.00], ubg=[inf, inf, 0])
    x_opt = r['x']
    print('x_opt: ', x_opt)
    result = vector_to_matrix(np.array(x_opt).reshape(len(p0)))
    result = result / np.linalg.det(result)
    print("the result is", result)
    # sp.io.savemat('P_python.mat', {'P': result})
    print("the eigen_value of P is", np.linalg.eigvals(result))
    print('the target value finally will be ', my_check_function(Data, np.array(x_opt).reshape(len(p0))))
    return result


def object_function(P, x, xd, w):
    J_total = 0
    for i in np.arange(len(x[0])):
        dlyap_dx, dlyap_dt = compute_Energy_Single(x[:, i], xd[:, i], P)
        # norm_vx = sqrt(dlyap_dx[0]**2 + dlyap_dx[1]**2 + dlyap_dx[2] ** 2)
        # norm_xd = sqrt(xd[:, i][0]**2 + xd[:, i][1]**2 + xd[:, i][2]**2)
        if len(x) == 3:
            norm_vx = sqrt(dlyap_dx[0]**2 + dlyap_dx[1]**2 + dlyap_dx[2] ** 2)
            norm_xd = sqrt(xd[:, i][0]**2 + xd[:, i][1]**2 + xd[:, i][2]**2)
        else:
            norm_vx = sqrt(dlyap_dx[0]**2 + dlyap_dx[1]**2)
            norm_xd = sqrt(xd[:, i][0]**2 + xd[:, i][1]**2)

        J = if_else(logic_or(norm_xd == 0, norm_vx == 0), 0, dlyap_dt / (norm_vx * norm_xd))
        J_total += if_else(dlyap_dt < 0, -w * J**2, J ** 2)
        # if norm_xd == 0 or norm_vx == 0:
        #     J = 0
        # else:
        #     J = dlyap_dt / (norm_vx * norm_xd)
        #     if dlyap_dt < 0:
        #         J_total += -w * J**2
        #     else:
        #         J_total += J ** 2
    return J_total


def compute_Energy_Single(x, xd, p):
    # lyap resp to x (P + P.T) @ X : shape: 3
    if len(x) == 3:
        dlyap_dx_1 = 2 * (p[0] * x[0] + p[1] * x[1] + p[2] * x[2])
        dlyap_dx_2 = 2 * (p[1] * x[0] + p[3] * x[1] + p[4] * x[2])
        dlyap_dx_3 = 2 * (p[2] * x[0] + p[4] * x[1] + p[5] * x[2])
        # lyap resp to t
        v_dot = xd[0] * dlyap_dx_1 + xd[1] * dlyap_dx_2 + xd[2] * dlyap_dx_3
        # derivative of x
        dv = [dlyap_dx_1, dlyap_dx_2, dlyap_dx_3]
    else:
        dlyap_dx_1 = 2 * (p[0] * x[0] + p[1] * x[1])
        dlyap_dx_2 = 2 * (p[1] * x[0] + p[2] * x[1])
        v_dot = xd[0] * dlyap_dx_1 + xd[1] * dlyap_dx_2
        dv = [dlyap_dx_1, dlyap_dx_2]

    return dv, v_dot


def constrians(p):
    P = np.array([[p[0], p[1], p[2]], [p[1], p[3], p[4]],[p[2], p[4], p[5]]])
    return sp.linalg.eigvals(P)


def vector_to_matrix(p):
    if len(p) == 6:
        P = np.array([[p[0], p[1], p[2]], [p[1], p[3], p[4]], [p[2], p[4], p[5]]])
    else:
        P = np.array([[p[0], p[1]],
                      [p[1], p[2]]])
    return P


def calculate_eigenvalue(P):
    a, b, c, d, e, f = P[0], P[3], P[5], P[1], P[4], P[2]
    x_1 = a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c + 3 * (d ** 2 + f ** 2 + e ** 2)
    x_2 = (-(2 * a - b - c) * (2 * b - a - c) * (2 * c - a - b) + 9 * (
                (d ** 2) * (2 * c - a - b) + (f ** 2) * (2 * b - a - c) + (e ** 2) * (2 * a - b - c)) +
           - 54 * d * e * f)

    theta = if_else(x_2 > 0, atan(sqrt(4*x_1**3 - x_2**2) / x_2), if_else(x_2 < 0, atan(sqrt(4 * x_1 ** 3 - x_2 ** 2) / x_2) + pi, pi/2))

    lambda_1 = (a + b + c - 2 * sqrt(x_1) * cos(theta/3)) / 3
    lambda_2 = (a + b + c + 2 * sqrt(x_1) * cos((theta - pi) / 3)) / 3
    lambda_3 = (a + b + c + 2 * sqrt(x_1) * cos((theta + pi) / 3)) / 3

    return lambda_1, lambda_2, lambda_3


def calculate_eigenvalue_2D(P):
    a = P[0]
    b = P[2]
    c = P[1]
    gamma_ = sqrt(4*(c**2) + (a - b)**2)
    lambda_1 = (a + b - gamma_) / 2
    lambda_2 = (a + b + gamma_) / 2
    return lambda_1, lambda_2


def cov_initial_guess(data):
    cov = np.cov(data)
    print('previous eigenvalues is ', np.linalg.eigvals(cov))
    U, S, VT = np.linalg.svd(cov)
    S = S * 100  #expand the eigen value
    cov = U @ np.diag(S) @ VT
    print("After expansion is ", np.linalg.eigvals(cov))
    if len(data) == 3:
        p = [cov[0][0], cov[0][1], cov[0][2], cov[1][1], cov[1][2], cov[2][2]]
    else:
        p = [cov[0][0], cov[0][1], cov[1][1]]
    return p


