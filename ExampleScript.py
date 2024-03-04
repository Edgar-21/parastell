import parastell
import logging
import csv
import numpy as np
#### THIS GETS THE SMOOTHED RADIAL DISTANCE FROM SOL TO COILS AT EACH PHI,
# THETA PAIR. IN THE DIRECTORY MUST EXIST A COILS STEP FILE NAME 'coils.step'
# additionally it is recommended that num_phi/phi_smooth step should be an integer
# likewise with theta


# Define number of toroidal cross-sections to make
num_phi = 80
# Define number of poloidal points to include in each toroidal cross-section
num_theta = 90
# Define the number of points to step over in the poloidal direction for smoothing
theta_smooth_step = 3
# Define the number of points to step over in the toroidal direction for smoothing
phi_smooth_step = 5

# Define plasma equilibrium VMEC file
plas_eq = 'plas_eq.nc'
# Define radial build
build = {
    'phi_list': np.linspace(0,90,num_phi),
    'theta_list': np.linspace(0,360, num_theta),
    'wall_s': 1.2,
    'radial_build': {
    }
}
# Define number of periods in stellarator plasma
num_periods = 4
# Define number of periods to generate
gen_periods = 1



# Define magnet coil parameters
magnets = {
    'file': 'coils.txt',
    'cross_section': ['circle', 20],
    'start': 3,
    'stop': None,
    'sample': 6,
    'name': 'magnet_coils',
    'h5m_tag': 'magnets',
    'meshing': False
}
# Define source mesh parameters
source = {
    'num_s': 11,
    'num_theta': 81,
    'num_phi': 241
}
# Define export parameters
export = {
    'exclude': [],
    'graveyard': False,
    'step_export': True,
    'h5m_export': None,
    'plas_h5m_tag': 'Vacuum',
    'sol_h5m_tag': 'Vacuum',
    # Note the following export parameters are used only for Cubit H5M exports
    'facet_tol': 1,
    'len_tol': 5,
    'norm_tol': None,
    # Note the following export parameters are used only for Gmsh H5M exports
    'min_mesh_size': 5.0,
    'max_mesh_size': 20.0,
    'volume_atol': 0.00001,
    'center_atol': 0.00001,
    'bounding_box_atol': 0.00001
}

# Define logger. Note that this is identical to the default logger instatiated
# by log.py. If no logger is passed to parametric_stellarator, this is the
# logger that will be used.
logger = logging.getLogger('log')
# Configure base logger message level
logger.setLevel(logging.INFO)
# Configure stream handler
s_handler = logging.StreamHandler()
# Configure file handler
f_handler = logging.FileHandler('stellarator.log')
# Define and set logging format
format = logging.Formatter(
    fmt = '%(asctime)s: %(message)s',
    datefmt = '%H:%M:%S'
)
s_handler.setFormatter(format)
f_handler.setFormatter(format)
# Add handlers to logger
logger.addHandler(s_handler)
logger.addHandler(f_handler)

# Create stellarator
strengths, point_list = parastell.parastell(
    plas_eq, num_periods, build, gen_periods, num_phi, num_theta,
    magnets = None, source = None, get_plasma_points = True,
    export = export, logger = logger
)

# find available radial distance and apply smoothing algorithm
coil_path = 'magnet_coils.step'

radial_distances = parastell.get_radial_real_estate(point_list, coil_path, num_phi, num_theta)
radial_distances = parastell.smooth_torus(num_phi, num_theta, phi_smooth_step, theta_smooth_step, radial_distances, h=1, steps=25)

#write out radial_distances to csv to avoid needing to recalculate it
with open('radial_distances.csv', 'w') as csvFile:
        csvwriter = csv.writer(csvFile)
        csvwriter.writerows(radial_distances)

