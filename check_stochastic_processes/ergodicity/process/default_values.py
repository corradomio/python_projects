"""
This file contains the default values for the parameters of the process initialization and simulation.
The parameters are optimized here for the balance between speed and precision.
"""
import numpy as np
import os
import json

t_default = 10 # time horizon. It is the time at which the process stops
timestep_default = 0.01 # time step. It is the time increment
num_instances_default = 5 # number of instances of the process
num_ensembles_default = 10 # number of ensembles of the process
drift_term_default=0.0 # drift term
stochastic_term_default=1.0 # stochastic term
dim_default=10 # dimension of the process
function_default = lambda t: t # default function of time
X_default = 1   # initial value of the process
Y_default = 1   # initial value of the process when another dimension is needed
dX_default = 0.01 # increment of the process
dY_default = 0.01   # increment of the process when another dimension is needed
mask_default = 100  # mask for the process. It is the number of steps after which the data is visualized. It may be needed when the data in the initial timesteps is very unstable, such as with the moment calculation.
verbosity_step_default = 1000000    # verbosity step for the process. It is the number of steps after which the data is printed
mean_list_default = np.array([0.0, 0.0, 0.0])  # default mean values for a multivariate correlated process
variance_matrix_default = np.array([[1.0, 0.6, 0.3],
                                    [0.6, 1.0, 0.6],
                                    [0.3, 0.6, 1.0]])  # default variance matrix for a multivariate correlated process
correlation_matrix_default = np.array([[1, 0.6, 0.3],
                                       [0.6, 1, 0.6],
                                       [0.3, 0.6, 1]])  # default correlation matrix for a multivariate correlated process
alpha_default = 2   # default alpha value for the Levy process
beta_default = 0  # default beta value for the Levy process
scale_default = (1/2)**0.5  # default scale value for the Levy process. It is set to 1/sqrt(2) to make the variance of the process equal to 1 when alpha=2 and the process is thus equal to Brownian motion
# scale_default = 1
loc_default = 0 # default location value for the Levy process
hurst_default = 0.5 # default Hurst exponent for the fractional Brownian motion
output_dir_default = "output"   # default output directory for the process
variances_default = np.array([1, 1, 1])  # default variances for a multivariate correlated process

# Path to the user's config file
user_config_path = os.path.join(os.getcwd(), 'default_values.json')

# Function to update default config with user-provided values
def load_user_config():
    if os.path.exists(user_config_path):
        try:
            with open(user_config_path, 'r') as file:
                user_config = json.load(file)

                # Update global variables if keys are present in user config
                globals().update({
                    key: value
                    for key, value in user_config.items()
                    if key in globals()  # Only update known config keys
                })

        except json.JSONDecodeError:
            print("Error: Invalid JSON format in default_values.json. Using default settings.")
    else:
        print(f"No user configuration file found at {user_config_path}. Using default settings.")

# Load the user config if it exists
load_user_config()
