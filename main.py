import numpy as np
import json, os


from utils_ds import load_tools


# import utilities
from utils_ds.rearrange_clusters import rearrange_clusters
from utils_ds.structures import ds_plot_options

# import optimization related method
from utils_ds.optimization_tool.optimize_P import optimize_P
from utils_ds.optimization_tool.optimize_A_b import optimize_lpv_ds_from_data

# import math related function
from utils_ds.math_tool.ds_related.compute_metrics import reproduction_metrics
from utils_ds.math_tool.ds_related.lpv_ds_function import lpv_ds

# import plotting related function
from utils_ds.plotting_tool.VisualizeEstimatedDS import VisualizeEstimatedDS
from utils_ds.plotting_tool.plot_lyapunov_and_derivatives import plot_lyapunov_and_derivatives


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def extract_param(data):
    K = data['K']
    M = data['M']
    Priors = np.array(data['Priors'])
    Mu = np.array(data['Mu']).reshape(K, -1)
    Sigma = np.array(data['Sigma']).reshape(K, M, M)
    # att = np.array(data['attractor'])
    return K, M, Priors, Mu, Sigma


def write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def read_data(data):
    return data["Data"], data["Data_sh"], data["att"], data["x0_all"], data["dt"], data["traj_length"]


class DsOpt:
    def __init__(self, data, js_path):
        self.Data, self.Data_sh, self.att, self.x0_all, self.dt, self.traj_length = read_data(data)
        self.js_path = js_path
        self.original_js = read_json(js_path)
        self.K, self.M, self.Priors, self.Mu, self.Sigma = extract_param(self.original_js)
        print('read Mu is: ', self.Mu)
        self.ds_struct = rearrange_clusters(self.Priors, self.Mu, self.Sigma, self.att)
        # learned ds parameters
        self.A_k = np.zeros((self.K, self.M, self.M))
        self.b_k = np.zeros((self.M, self.K))
        self.P_opt = np.zeros((self.M, self.M))

    def begin(self):
        P_opt = optimize_P(self.Data_sh)
        A_k, b_k = optimize_lpv_ds_from_data(self.Data, self.att, 2, self.ds_struct, P_opt, 0)
        
        # document the learned ds
        self.A_k = A_k
        self.b_k = b_k
        self.P_opt = P_opt

        new_A_k = np.copy(self.A_k)
        new_Sig = np.copy(self.Sigma)

        # convert in-order to the ros data recovery
        for k in range(self.K):
            new_A_k[k] = new_A_k[k].T
            new_Sig[k] = new_Sig[k].T
        Mu_trans = self.ds_struct.Mu.T

        new_A_k = new_A_k.reshape(-1).tolist()
        self.original_js['Sigma'] = new_Sig.reshape(-1).tolist()
        self.original_js['Mu'] = Mu_trans.reshape(-1).tolist()
        self.original_js['Prior'] = self.ds_struct.Priors.tolist()
        self.original_js['A'] = new_A_k
        self.original_js['attractor']= self.att.ravel().tolist()
        self.original_js['att_all']= self.att.ravel().tolist()
        self.original_js["dt"] = self.dt
        self.original_js["gripper_open"] = 0

        # x

        # (Data, A_k, b_k, traj_length, x0_all, ds_struct)
        write_json(self.original_js, self.js_path)

    def evaluate(self):
        rmse, e_dot, dwtd = reproduction_metrics(self.Data, self.A_k, self.b_k,
                                                 self.traj_length, self.x0_all, self.ds_struct)
        print("the reproduced RMSE is ", rmse)
        print("the reproduced e_dot is", e_dot)
        print("the reproduced dwtd is ", dwtd)

    def make_plot(self):
        Data_dim = self.M
        ds_handle = lambda x_velo: lpv_ds(x_velo, self.ds_struct, self.A_k, self.b_k)
        ds_opt_plot_option = ds_plot_options()
        ds_opt_plot_option.x0_all = self.x0_all

        # The plotting function for lyapunov only valid for data with 2 dimension
        if Data_dim == 2:
            plot_lyapunov_and_derivatives(self.Data, ds_handle, self.att, self.P_opt)

        # Visualized the reproduced trajectories
        VisualizeEstimatedDS(self.Data[:Data_dim], ds_handle, ds_opt_plot_option)


if __name__ == '__main__':
    pkg_dir = os.path.join(os.getcwd(), 'data')
    chosen_dataset = 4  # 6 # 4 (when conducting 2D test)
    sub_sample = 2  # '>2' for real 3D Datasets, '1' for 2D toy datasets
    nb_trajectories = 4  # Only for real 3D data
    Data, Data_sh, att, x0_all, _, dt, traj_length = load_tools.load_dataset_DS(pkg_dir, chosen_dataset, sub_sample,
                                                                        nb_trajectories)
    data = {
        "Data": Data,
        "Data_sh": Data_sh,
        "att": att,
        "x0_all": x0_all,
        "dt": dt,
        "traj_length":traj_length
    }

    ds_opt = DsOpt(data, os.path.join(pkg_dir, "output.json"))
    ds_opt.begin()
    ds_opt.evaluate()
    ds_opt.make_plot()
