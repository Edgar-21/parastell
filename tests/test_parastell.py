from pathlib import Path

import numpy as np
import pytest

import parastell.parastell as ps
from parastell.cubit_io import create_new_cubit_instance

files_to_remove = [
    "chamber.step",
    "component.step",
    "magnet_set.step",
    "magnet_mesh.exo",
    "magnet_mesh.h5m",
    "dagmc.h5m",
    "dagmc.cub5",
    "source_mesh.h5m",
    "stellarator.log",
    "step_import.log",
    "step_export.log",
]


def remove_files():
    for file in files_to_remove:
        if Path(file).exists():
            Path.unlink(file)


@pytest.fixture
def stellarator():

    vmec_file = Path("files_for_tests") / "wout_vmec.nc"

    stellarator_obj = ps.Stellarator(vmec_file)

    return stellarator_obj


def test_parastell(stellarator):

    remove_files()
    create_new_cubit_instance()
    # In-Vessel Build

    toroidal_angles = [0.0, 5.0, 10.0, 15.0]
    poloidal_angles = [0.0, 120.0, 240.0, 360.0]
    wall_s = 1.08
    component_name_exp = "component"
    radial_build_dict = {
        component_name_exp: {
            "thickness_matrix": np.ones(
                (len(toroidal_angles), len(poloidal_angles))
            )
            * 10
        }
    }
    num_ribs = 11

    stellarator.construct_invessel_build(
        toroidal_angles,
        poloidal_angles,
        wall_s,
        radial_build_dict,
        num_ribs=num_ribs,
    )

    chamber_filename_exp = Path("chamber").with_suffix(".step")
    component_filename_exp = Path(component_name_exp).with_suffix(".step")

    stellarator.export_invessel_build()

    assert chamber_filename_exp.exists()
    assert component_filename_exp.exists()

    # Magnet Coils

    coils_file = Path("files_for_tests") / "coils.example"
    width = 40.0
    thickness = 50.0
    toroidal_extent = 90.0
    sample_mod = 6

    stellarator.construct_magnets_from_filaments(
        coils_file, width, thickness, toroidal_extent, sample_mod=sample_mod
    )

    step_filename_exp = "magnet_set.step"
    export_mesh = True
    mesh_filename_exp = "magnet_mesh"

    stellarator.export_magnets(
        step_filename=step_filename_exp,
        export_mesh=export_mesh,
        mesh_filename=mesh_filename_exp,
    )

    assert Path(step_filename_exp).with_suffix(".step").exists()
    assert Path(mesh_filename_exp).with_suffix(".h5m").exists()

    mesh_size = (6, 41, 9)
    toroidal_extent = 15.0

    stellarator.construct_source_mesh(mesh_size, toroidal_extent)

    filename_exp = "source_mesh"

    stellarator.export_source_mesh(filename=filename_exp)

    assert Path(filename_exp).with_suffix(".h5m").exists()

    chamber_volume_id_exp = 1
    component_volume_id_exp = 2
    magnet_volume_ids_exp = list(range(3, 4))
    filename_exp = "dagmc"

    stellarator.build_cubit_model()

    assert (
        stellarator.invessel_build.radial_build.radial_build["chamber"][
            "vol_id"
        ]
        == chamber_volume_id_exp
    )
    assert (
        stellarator.invessel_build.radial_build.radial_build[
            component_name_exp
        ]["vol_id"]
        == component_volume_id_exp
    )
    assert stellarator.magnet_set.volume_ids == magnet_volume_ids_exp

    stellarator.export_cubit_dagmc(filename=filename_exp)
    stellarator.export_cub5(filename=filename_exp)

    assert Path(filename_exp).with_suffix(".h5m").exists()
    assert Path(filename_exp).with_suffix(".cub5").exists()

    remove_files()

    stellarator.build_cad_to_dagmc_model()
    stellarator.export_cad_to_dagmc(min_mesh_size=50, max_mesh_size=100)

    assert Path(filename_exp).with_suffix(".h5m").exists()
    # Test with custom magnet geometry

    create_new_cubit_instance()

    geometry_file = Path("files_for_tests") / "magnet_geom.step"

    stellarator.construct_invessel_build(
        toroidal_angles,
        poloidal_angles,
        wall_s,
        radial_build_dict,
        num_ribs=num_ribs,
    )

    stellarator.export_invessel_build()

    stellarator.add_magnets_from_geometry(geometry_file)

    stellarator.build_cubit_model()

    assert (
        stellarator.invessel_build.radial_build.radial_build["chamber"][
            "vol_id"
        ]
        == chamber_volume_id_exp
    )
    assert (
        stellarator.invessel_build.radial_build.radial_build[
            component_name_exp
        ]["vol_id"]
        == component_volume_id_exp
    )
    assert stellarator.magnet_set.volume_ids == magnet_volume_ids_exp

    stellarator.export_cubit_dagmc(filename=filename_exp)
    stellarator.export_cub5(filename=filename_exp)

    assert Path(filename_exp).with_suffix(".h5m").exists()
    assert Path(filename_exp).with_suffix(".cub5").exists()

    remove_files()

    stellarator.build_cad_to_dagmc_model()
    stellarator.export_cad_to_dagmc(min_mesh_size=50, max_mesh_size=100)

    assert Path(filename_exp).with_suffix(".h5m").exists()

    remove_files()

    # Test pydagmc model generation

    num_surfaces_exp = 8
    num_volumes_exp = 2

    stellarator.construct_invessel_build(
        toroidal_angles,
        poloidal_angles,
        wall_s,
        radial_build_dict,
        num_ribs=num_ribs,
        use_pydagmc=True,
    )

    stellarator.build_cubit_model()
    stellarator.export_cubit_dagmc()
    stellarator.merge_magnet_and_ivb_dagmc_models()

    assert len(stellarator.merged_model.surfaces) == num_surfaces_exp
    assert len(stellarator.merged_model.volumes) == num_volumes_exp

    stellarator.merged_model.write_file("dagmc.h5m")

    assert Path("dagmc.h5m").exists()

    remove_files()
