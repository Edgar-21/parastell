import parastell.nwl_utils as nwl


# Define simulation parameters
dagmc_geom = "nwl_geom.h5m"
source_mesh = "source_mesh.h5m"
tor_ext = 90.0
ss_file = "source_strengths.txt"
num_parts = 100000

nwl.nwl_transport(dagmc_geom, source_mesh, tor_ext, ss_file, num_parts)

# Define first wall geometry and plotting parameters
source_file = "surface_source.h5"
plas_eq = "wout_vmec.nc"
tor_ext = 90.0
pol_ext = 360.0
wall_s = 1.08

crossings = nwl.extract_coords(source_file)
source_strength = sum(nwl.extract_ss(ss_file))
nwl_mat, area_array, phi_pts, theta_pts, bin_arr = (
    nwl.extract_nwl_from_surface_crossings(
        crossings,
        source_strength,
        plas_eq,
        tor_ext,
        pol_ext,
        wall_s,
        num_phi=30,
        num_theta=25,
        num_threads=10,
    )
)
nwl.plot_nwl(nwl_mat, phi_pts, theta_pts, 10, filename="test_nwl.png")
nwl.write_nwl_to_mesh(nwl_mat, bin_arr)
