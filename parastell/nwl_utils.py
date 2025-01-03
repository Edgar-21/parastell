import h5py
import numpy as np
from scipy.optimize import direct
import openmc
import pystell.read_vmec as read_vmec
import matplotlib.pyplot as plt
import concurrent.futures
import math
import matplotlib
from pymoab import core, types
from parastell.utils import dt_neutron_energy_ev, eV2J, J2MJ

matplotlib.use("agg")


def extract_ss(ss_file):
    """Extracts list of source strengths for each tetrahedron from input file.

    Arguments:
        ss_file (str): path to source strength input file.

    Returns:
        strengths (list): list of source strengths for each tetrahedron (1/s).
            Returned only if source mesh is generated.
    """
    strengths = []

    file_obj = open(ss_file, "r")
    data = file_obj.readlines()
    for line in data:
        strengths.append(float(line))

    return strengths


def nwl_transport(dagmc_geom, source_mesh, tor_ext, ss_file, num_parts):
    """Performs neutron transport on first wall geometry via OpenMC. The
    first wall must be tagged as a vacuum boundary during the creating of the
    DAGMC geometry.

    Arguments:
        dagmc_geom (str): path to DAGMC geometry file.
        source_mesh (str): path to source mesh file.
        tor_ext (float): toroidal extent of model (deg).
        ss_file (str): source strength input file.
        num_parts (int): number of source particles to simulate.
    """
    tor_ext = np.deg2rad(tor_ext)

    strengths = extract_ss(ss_file)

    # Initialize OpenMC model
    model = openmc.model.Model()

    dag_univ = openmc.DAGMCUniverse(dagmc_geom, auto_geom_ids=False)

    # Define problem boundaries
    per_init = openmc.YPlane(boundary_type="periodic", surface_id=9991)
    per_fin = openmc.Plane(
        a=np.sin(tor_ext),
        b=-np.cos(tor_ext),
        c=0,
        d=0,
        boundary_type="periodic",
        surface_id=9990,
    )

    # Define first period of geometry
    region = +per_init & +per_fin
    period = openmc.Cell(cell_id=9996, region=region, fill=dag_univ)
    geometry = openmc.Geometry([period])
    model.geometry = geometry

    # Define run settings
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = num_parts
    settings.batches = 1

    # Define neutron source
    mesh = openmc.UnstructuredMesh(source_mesh, "moab")
    src = openmc.IndependentSource()
    src.space = openmc.stats.MeshSpatial(
        mesh, strengths=strengths, volume_normalized=False
    )
    src.angle = openmc.stats.Isotropic()
    src.energy = openmc.stats.Discrete([14.1e6], [1.0])
    settings.source = [src]

    # Track surface crossings
    settings.surf_source_write = {
        "surface_ids": [1],
        "max_particles": num_parts * 2,
    }

    model.settings = settings

    model.run()


def min_problem(theta, vmec, wall_s, phi, pt):
    """Minimization problem to solve for the poloidal angle.

    Arguments:
        theta (float): poloidal angle (rad).
        vmec (object): plasma equilibrium object.
        wall_s (float): closed flux surface label extrapolation at wall.
        phi (float): toroidal angle (rad).
        pt (array of float): Cartesian coordinates of interest (cm).

    Returns:
        diff (float): L2 norm of difference between coordinates of interest and
            computed point (cm).
    """
    # Compute first wall point
    fw_pt = np.array(vmec.vmec2xyz(wall_s, theta, phi))
    m2cm = 100
    fw_pt = fw_pt * m2cm

    diff = np.linalg.norm(pt - fw_pt)

    return diff


def find_coords(data):
    """Solves for poloidal angle of plasma equilibrium corresponding to
    specified Cartesian coordinates. Takes a single arg so it works nicely with
    ProcessPoolExecutor

    Arguments:
        data (tuple of (str, float, list of tuple of float)): First element is
            the path to the plasma equilibrium file, second is the wall_s value,
            3rd is the list of phi, xyz coordinate pairs to solve for theta at.

    Returns:
        thetas (list of float): poloidal angles (rad) corresponding to phi xyz
            coordinate pairs.
    """
    thetas = []
    vmec = read_vmec.VMECData(data[0])
    wall_s = data[1]
    phi_xyz_coords = data[2]

    for coords in phi_xyz_coords:
        theta = direct(
            min_problem,
            bounds=[(-np.pi, np.pi)],
            args=(vmec, wall_s, coords[0], coords[1]),
        )
        # remap from [-pi,pi] to [0, 2pi] to match parastell's angle convention
        thetas.append((theta.x[0] + 2 * np.pi) % (2 * np.pi))
    return thetas


def flux_coords(plas_eq, wall_s, coords, num_threads):
    """Computes flux-coordinate toroidal and poloidal angles corresponding to
    specified Cartesian coordinates.

    Arguments:
        vmec (object): plasma equilibrium object.
        wall_s (float): closed flux surface label extrapolation at wall.
        coords (array of array of float): Cartesian coordinates of all particle
            surface crossings (cm).

    Returns:
        phi_coords (array of float): toroidal angles of surface crossings (rad).
        theta_coords (array of float): poloidal angles of surface crossings
            (rad).
    """

    phi_coords = np.arctan2(coords[:, 1], coords[:, 0])
    chunk_size = math.ceil(len(phi_coords) / num_threads)
    chunks = []
    for i in range(num_threads):
        chunk = list(
            zip(
                phi_coords[i * chunk_size : (i + 1) * chunk_size],
                coords[i * chunk_size : (i + 1) * chunk_size],
            )
        )
        chunks.append((plas_eq, wall_s, chunk))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_threads
    ) as executor:
        theta_coord_chunks = list(executor.map(find_coords, chunks))
        theta_coords = [
            theta_coord
            for theta_coord_chunk in theta_coord_chunks
            for theta_coord in theta_coord_chunk
        ]

    return phi_coords.tolist(), theta_coords


def extract_coords(source_file):
    """Extracts Cartesian coordinates of particle surface crossings given an
    OpenMC surface source file.

    Arguments:
        source_file (str): path to OpenMC surface source file.

    Returns:
        coords (array of array of float): Cartesian coordinates of all particle
            surface crossings.
    """
    # Load source file
    file = h5py.File(source_file, "r")
    # Extract source information
    dataset = file["source_bank"]["r"]
    # Construct matrix of particle crossing coordinates
    coords = np.empty((len(dataset), 3))
    coords[:, 0] = dataset["x"]
    coords[:, 1] = dataset["y"]
    coords[:, 2] = dataset["z"]

    return coords


def plot_nwl(
    nwl_mat, phi_pts, theta_pts, num_levels, title="NWL", filename="NWL.png"
):
    """Generates contour plot of NWL.

    Arguments:
        nwl_mat (array of array of float): NWL solutions at centroids of
            (phi, theta) bins (MW).
        phi_pts (array of float): centroids of toroidal angle bins (rad).
        theta_bins (array of float): centroids of poloidal angle bins (rad).
        num_levels (int): number of contour regions.
        title (str): plot title, default "NWL"
        filename (str): path to save the figure to. Defaults to "NWL.png" in
            the cwd.
    """
    phi_pts = np.rad2deg(phi_pts)
    theta_pts = np.rad2deg(theta_pts)

    levels = np.linspace(np.min(nwl_mat), np.max(nwl_mat), num=num_levels)
    fig, ax = plt.subplots()
    CF = ax.contourf(phi_pts, theta_pts, nwl_mat.T, levels=levels)
    cbar = plt.colorbar(CF)
    cbar.ax.set_ylabel("NWL (MW/m2)")
    plt.xlabel("Toroidal Angle (degrees)")
    plt.ylabel("Poloidal Angle (degrees)")
    plt.title(title)
    fig.savefig(filename)


def area_from_corners(corners):
    """
    Calculate an approximation of the area defined by 4 xyz points

    Arguments:
        corners (4x3 numpy array): list of 4 (x,y,z) points. Connecting the
            points in the order given should result in a polygon

    Returns:
        area (float): approximation of area
    """
    # triangle 1
    v1 = corners[3] - corners[0]
    v2 = corners[2] - corners[0]

    v3 = np.cross(v1, v2)

    area1 = np.sqrt(np.sum(np.square(v3))) / 2

    # triangle 2
    v1 = corners[1] - corners[0]
    v2 = corners[2] - corners[0]

    v3 = np.cross(v1, v2)

    area2 = np.sqrt(np.sum(np.square(v3))) / 2

    area = area1 + area2

    return area


def create_moab_tris_from_corners(corners, mbc):
    """Create 2 moab triangle elements from a list of 4 x y z points.

    Arguments:
        corners (4x3 numpy array): list of 4 (x,y,z) points. Connecting the
            points in the order given should result in a polygon
        mbc (pymoab core): pymoab core instance to create elements with.

    Returns:
        tri_1 (pymoab element): Triangular mesh element.
        tri_2 (pymoab element): Triangular mesh element.
    """
    tri_1_verts = mbc.create_vertices([corners[0], corners[1], corners[2]])
    tri_2_verts = mbc.create_vertices([corners[3], corners[2], corners[0]])

    tri_1 = mbc.create_element(types.MBTRI, tri_1_verts)
    tri_2 = mbc.create_element(types.MBTRI, tri_2_verts)

    return tri_1, tri_2


def write_nwl_to_mesh(
    nwl_mat, bin_arr, tag_name="NWL", filename="nwl_mesh.vtk"
):
    """Use pymoab to export NWL data to a mesh for visualization.

    Arguments:
        nwl_mat (numpy array): NxM array of NWL values.
        bin_arr (numpy array): NxMx3 array of edges of bins in xyz space.
        tag_name (str): Tag for the NWL data on the mesh. Default "NWL"
        filename (str): Path to save the mesh to. Default "nwl_mesh.vtk" in the
            cwd.
    """
    mbc = core.Core()
    mb_tris = []
    mb_data = []
    for phi_index, theta_index in np.ndindex(nwl_mat.shape):
        nwl = nwl_mat[phi_index, theta_index]
        corner1 = bin_arr[phi_index, theta_index]
        corner2 = bin_arr[phi_index, theta_index + 1]
        corner3 = bin_arr[phi_index + 1, theta_index + 1]
        corner4 = bin_arr[phi_index + 1, theta_index]
        corners = [corner1, corner2, corner3, corner4]
        tri_1, tri_2 = create_moab_tris_from_corners(corners, mbc)
        mb_tris += [tri_1, tri_2]
        mb_data += [nwl, nwl]

    tag_handle = mbc.tag_get_handle(
        tag_name,
        size=1,
        tag_type=types.MB_TYPE_DOUBLE,
        storage_type=types.MB_TAG_DENSE,
        create_if_missing=True,
    )
    mbc.tag_set_data(tag_handle, mb_tris, mb_data)
    mbc.write_file(filename)


def extract_nwl_from_surface_crossings(
    crossings,
    source_strength,
    plas_eq,
    tor_ext,
    pol_ext,
    wall_s,
    num_phi=101,
    num_theta=101,
    chunk_size=None,
    num_threads=1,
):
    """Bins a list of xyz coordinates onto a 2-D histogram in poloidal,
    toroidal space.

    Arguments:
        crossings (iterable of iterable of float): Iterable of x, y, z surface
            crossing coordinates.
        source_strength (float): Source strength, in neutrons/second.
        plas_eq (str): path to plasma equilibrium NetCDF file.
        tor_ext (float): toroidal extent of model (deg).
        pol_ext (float): poloidal extent of model (deg).
        wall_s (float): closed flux surface label extrapolation at wall.
        num_phi (int): number of toroidal angle bins (defaults to 101).
        num_theta (int): number of poloidal angle bins (defaults to 101).
        chunk_size (int): number of crossings to calculate at once, to help
            with potential memory limits. If None all crossings will be done
            at once
        num_threads (int): number of threads to use for NWL calculations,
            defaults to 1.

    Returns:
        nwl_mat (numpy array): NxM array of NWL values for each bin in MW/m2.
        area_array (numpy array): NxM array of area values for each bin in m2.
        phi_pts (numpy array): Centroids of bins in toroidal direction in
            radians.
        theta_pts (numpy array): Centroids of bins in poloidal direction in
            radians.
        bin_arr (numpy array): NxMx3 array of edges of bins in xyz space in m.
    """
    tor_ext = np.deg2rad(tor_ext)
    pol_ext = np.deg2rad(pol_ext)

    vmec = read_vmec.VMECData(plas_eq)

    phi_coords = []
    theta_coords = []

    if chunk_size is None:
        chunk_size = len(crossings)

    chunks = math.ceil(len(crossings) / chunk_size)

    for i in range(chunks):
        phi_coord_subset, theta_coord_subset = flux_coords(
            plas_eq,
            wall_s,
            crossings[i * chunk_size : (i + 1) * chunk_size],
            num_threads,
        )
        phi_coords += phi_coord_subset
        theta_coords += theta_coord_subset

    # Define minimum and maximum bin edges for each dimension
    phi_min = 0 - tor_ext / num_phi / 2
    phi_max = tor_ext + tor_ext / num_phi / 2

    theta_min = 0 - pol_ext / num_theta / 2
    theta_max = pol_ext + pol_ext / num_theta / 2

    # Bin particle crossings
    count_mat, phi_bins, theta_bins = np.histogram2d(
        phi_coords,
        theta_coords,
        bins=[num_phi, num_theta],
        range=[[phi_min, phi_max], [theta_min, theta_max]],
    )

    # adjust endpoints to eliminate overlap
    phi_bins[0] = 0
    phi_bins[-1] = tor_ext
    theta_bins[0] = 0
    theta_bins[-1] = pol_ext

    # Compute centroids of bin dimensions
    phi_pts = np.linspace(0, tor_ext, num=num_phi)
    theta_pts = np.linspace(0, pol_ext, num=num_theta)

    nwl_mat = (
        count_mat
        * dt_neutron_energy_ev
        * eV2J
        * source_strength
        * J2MJ
        / len(crossings)
    )

    # construct array of bin boundaries
    bin_arr = np.zeros((num_phi + 1, num_theta + 1, 3))

    for phi_bin, phi in enumerate(phi_bins):
        for theta_bin, theta in enumerate(theta_bins):
            x, y, z = vmec.vmec2xyz(wall_s, theta, phi)
            bin_arr[phi_bin, theta_bin, :] = [x, y, z]

    # construct area array
    area_array = np.zeros((num_phi, num_theta))

    for phi_index in range(num_phi):
        for theta_index in range(num_theta):
            # each bin has 4 (x,y,z) corners
            corner1 = bin_arr[phi_index, theta_index]
            corner2 = bin_arr[phi_index, theta_index + 1]
            corner3 = bin_arr[phi_index + 1, theta_index + 1]
            corner4 = bin_arr[phi_index + 1, theta_index]
            corners = np.array([corner1, corner2, corner3, corner4])
            area = area_from_corners(corners)
            area_array[phi_index, theta_index] = area

    nwl_mat = nwl_mat / area_array

    return nwl_mat, area_array, phi_pts, theta_pts, bin_arr
