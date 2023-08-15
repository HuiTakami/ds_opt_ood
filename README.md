# LPV-DS OOD Style for DAMM Pipeline

This module is an enhanced rendition of the optimization section from:
https://github.com/penn-figueroa-lab/ds-opt-py

It has been adapted for a comprehensive real robot testing pipeline available at:
https://github.com/SunannnSun/damm_lpvds

Building upon the output from:
https://github.com/SunannnSun/damm,
this module learns a quadratic Lyapunov function. Subsequently, it undertakes regression within the bounds of the Lyapunov function to ensure both precision and stability.

---
### Plug in
To utilize it on your algorithm:

The data should be formulated as a dictionary:
```
data = {
    "Data": Data,         # Dimension x Number of Datapoints
    "Data_sh": Data_sh,   # Shifted attractor to 0 for 'Data' (used in Lyapunov function learning)
    "att": att,           # Dimension x 1
    "x0_all": x0_all,     # Dimension x Number of demonstrated trajectory
                          # This is the start points for all demonstrated trajectories
    "dt": dt,             # Sample time
}
```

For the clustering result, save this dictionary as a json file:
```
json_output = {
    "name": "Clustering Result",
    "K": # Number of clusters,
    "M": # Dimension,
    "Priors": # List of Prior,
    "Mu": # ravel K x M shape Mu to a list,
    "Sigma": #ravel K x M x M Sigma to a list,
}
```
---
### Usage
import DsOpt class and initialize the object:
```
ds_opt = DsOpt(#Your data dictionary, #Your json output directory)
```

Train:
```
ds_opt.begin()
```

Evaluate:
Return the rmse, e_dot, dwtd for learned trajectory
```
ds_opt.evaluate()
```

Plot: Make plots for Lyapunov derivative, value, and reproduced streamlines
```
ds_opt.evaluate()
```
![21901692136831_ pic](https://github.com/HuiTakami/ds_opt_ood/assets/97799818/c08f82e0-7837-4e43-a94f-072723a95f2c)
![21911692136852_ pic](https://github.com/HuiTakami/ds_opt_ood/assets/97799818/df892461-d154-4ca9-a2a2-5e8134606071)

![21921692136877_ pic_hd](https://github.com/HuiTakami/ds_opt_ood/assets/97799818/f2c9839b-8aa0-40ab-b142-648d9321c4be)

