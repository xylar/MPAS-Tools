#!/usr/bin/env python
"""
Interpolate the ISMIP7 Antarctic masks onto a MALI mesh.

The ISMIP7 melt-module calibration protocol (Reese et al.) aggregates modelled
melt over IMBIE2 drainage basins, over bins of equal buttressing importance
(BFRN), and over the Pine Island and Dotson ice shelves.  Those masks are
distributed on the ISMIP polar-stereographic grid; this script puts them on a
MALI mesh so that the same aggregation can be done in the ice-sheet model's own
discretisation.

Remapping is nearest-neighbour throughout, since every field is categorical.

Basin numbering
---------------
Two conventions are in play and they differ by one, which is easy to get wrong
and produces plausible-looking but incorrect basin aggregates:

* **ISMIP7** ``basin_numbers_ismip8km_v2.nc`` is 0-based, 0-15.  Basin 9 is the
  Eastern Amundsen (Pine Island, Dotson) and basin 14 is Ronne-Filchner, which
  is how the protocol refers to them.
* **MALI** ``ismip6shelfMelt_basin`` is 1-based, 1-16, being the
  ``regionCellMasks`` column index plus one (see ``tune_ismip6_melt_deltat.py``).

This script writes *both*, under distinct names, so neither is silently
reinterpreted:

* ``ismip7BasinNumber`` -- 0-based, for aggregation against ISMIP7 targets
* ``ismip6shelfMelt_basin`` -- 1-based, for MALI's melt-parameterisation input

If a MALI region-mask file is supplied with ``--region_mask``, the script
cross-tabulates the two and reports per-basin agreement, so a mismatch between
the ISMIP6-era regions and the ISMIP7 IMBIE2 basins is caught here rather than
downstream.

Example
-------
::

    python interpolate_ismip7_masks_to_mali.py \\
        -i ais_4to20km.20250625.nc \\
        --ismip7_dir /path/to/ISMIP7/AIS/parameterisations/ocean \\
        --region_mask ais_4to20km_region_mask.20230105.nc \\
        -o ais_4to20km_ismip7_masks.nc -n 128
"""

import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime, timezone

import numpy as np
import xarray as xr
from mpas_tools.logging import LoggingContext
from pyremap import Remapper

# EPSG:3031, the ISMIP Antarctic polar-stereographic projection
ISMIP_PROJ_STR = (
    '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 '
    '+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
)

# ice-shelf ids in shelf_mask_ismip8km.nc
PIG_ID = 110
DOTSON_ID = 97

# Pine Island is cut at this x to keep only its main trunk, following the
# protocol's worked example
PIG_X_MAX = -1.625e6

# codes written to ismip7ShelfRegion
REGION_CODES = {'none': 0, 'pig': 1, 'dotson': 2}


def parse_args(argv=None):
    parser = ArgumentParser(
        prog='interpolate_ismip7_masks_to_mali.py',
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-i', '--mali_mesh', dest='mali_mesh', required=True,
        metavar='FILENAME',
        help='MALI mesh file to interpolate onto',
    )
    parser.add_argument(
        '--ismip7_dir', dest='ismip7_dir', required=True, metavar='DIR',
        help='ISMIP7 parameterisations/ocean directory, containing imbie2/, '
             'bfrns/, floatingmasks/ and shelfmask/',
    )
    parser.add_argument(
        '-o', '--out_file', dest='out_file',
        default='ismip7_masks_on_mali.nc', metavar='FILENAME',
        help='output file (default: %(default)s)',
    )
    parser.add_argument(
        '-r', '--resolution', dest='resolution', type=int, default=8,
        help='ISMIP grid resolution in km (default: %(default)s)',
    )
    parser.add_argument(
        '-n', '--ntasks', dest='ntasks', type=int, default=1,
        help='number of MPI tasks for ESMF_RegridWeightGen '
             '(default: %(default)s)',
    )
    parser.add_argument(
        '-m', '--method', dest='method', default='neareststod',
        choices=['neareststod', 'bilinear', 'conserve'],
        help='remapping method; the masks are categorical so nearest '
             'neighbour is the appropriate choice (default: %(default)s)',
    )
    parser.add_argument(
        '--mapping_file', dest='mapping_file', default=None,
        metavar='FILENAME',
        help='mapping file to use or create; by default a name is derived '
             'from the mesh names and method',
    )
    parser.add_argument(
        '--region_mask', dest='region_mask', default=None, metavar='FILENAME',
        help='optional MALI region-mask file with regionCellMasks, used to '
             'cross-check the ISMIP7 basin numbering',
    )
    parser.add_argument(
        '--mali_mesh_name', dest='mali_mesh_name', default=None,
        help='name for the MALI mesh in the mapping file (default: derived '
             'from the mesh filename)',
    )
    return parser.parse_args(argv)


def load_ismip7_masks(ismip7_dir, resolution):
    """
    Assemble the ISMIP7 masks into one dataset on the ISMIP grid.

    Parameters
    ----------
    ismip7_dir : str
        The ``parameterisations/ocean`` directory of the ISMIP7 AIS datasets.
    resolution : int
        ISMIP grid resolution in km.

    Returns
    -------
    xarray.Dataset
        ``ismip7BasinNumber``, ``ismip7BFRNBin``, ``ismip7FloatingMask`` and
        ``ismip7ShelfRegion``, all on the ISMIP grid, as floats so that they
        can be remapped.
    """
    res = f'{resolution}km'

    basins = xr.open_dataset(
        os.path.join(ismip7_dir, 'imbie2', f'basin_numbers_ismip{res}_v2.nc')
    )['basinNumber']
    bfrn = xr.open_dataset(
        os.path.join(ismip7_dir, 'bfrns', f'BFRN_ismip{res}_v2.nc')
    )['BFRN_bins']
    floating = xr.open_dataset(
        os.path.join(
            ismip7_dir, 'floatingmasks', f'floatingmask_ismip{res}.nc'
        )
    )['mask']

    # the shelf mask is only distributed at 8 km; the PIG/Dotson regions are
    # small enough that remapping from 8 km is appropriate regardless
    shelves = xr.open_dataset(
        os.path.join(ismip7_dir, 'shelfmask', 'shelf_mask_ismip8km.nc')
    )['shelf_mask']
    if 'time' in shelves.dims:
        shelves = shelves.isel(time=0)

    pig = (shelves == PIG_ID) & (shelves['x'] > PIG_X_MAX)
    dotson = shelves == DOTSON_ID
    region = xr.where(
        pig, REGION_CODES['pig'],
        xr.where(dotson, REGION_CODES['dotson'], REGION_CODES['none']),
    )

    ds = xr.Dataset()
    ds['ismip7BasinNumber'] = basins.astype(float)
    ds['ismip7BFRNBin'] = bfrn.astype(float)
    ds['ismip7FloatingMask'] = floating.astype(float)
    ds['ismip7ShelfRegion'] = region.astype(float)
    return ds


def remap_to_mali(ds_masks, src_grid_file, mali_mesh, mali_mesh_name,
                  resolution, method, ntasks, mapping_file, logger):
    """
    Remap the ISMIP7 masks onto the MALI mesh with pyremap.

    Returns
    -------
    xarray.Dataset
        The masks on the MALI mesh, still as floats.
    """
    src_mesh_name = f'ismip{resolution}km'

    if mapping_file is None:
        mapping_file = (
            f'map_{src_mesh_name}_to_{mali_mesh_name}_{method}.nc'
        )

    remapper = Remapper(
        ntasks=ntasks, map_filename=mapping_file, method=method
    )
    remapper.src_from_proj(
        src_grid_file, src_mesh_name, proj_str=ISMIP_PROJ_STR
    )
    remapper.dst_from_mpas(mali_mesh, mali_mesh_name)

    # build_map() is what creates the source and destination descriptors, so
    # it has to be called even when the mapping file already exists; skipping
    # it leaves remap_numpy() with descriptors of None.  Weight generation for
    # nearest-neighbour is cheap, so simply rebuild.
    logger.info(f'building mapping file {mapping_file}')
    remapper.build_map(logger=logger)

    logger.info('remapping masks onto the MALI mesh')
    return remapper.remap_numpy(ds_masks)


def to_integer_masks(ds_remapped):
    """
    Round the remapped fields to integers and derive the MALI basin field.

    Nearest-neighbour remapping should already return exact source values, but
    they come back as floats; rounding makes the intent explicit and guards
    against a different method being used.
    """
    ds = xr.Dataset()

    basin0 = ds_remapped['ismip7BasinNumber']
    valid = basin0.notnull()
    basin0 = basin0.round().fillna(-1).astype(np.int32)

    ds['ismip7BasinNumber'] = basin0
    ds['ismip7BasinNumber'].attrs = {
        'long_name': 'IMBIE2 drainage basin number, ISMIP7 convention',
        'convention': '0-based, 0-15; basin 9 is Eastern Amundsen, '
                      'basin 14 is Ronne-Filchner',
        'valid_range': np.array([0, 15], dtype=np.int32),
    }

    # MALI's melt parameterisation expects the 1-based convention
    ds['ismip6shelfMelt_basin'] = xr.where(valid, basin0 + 1, 0).astype(
        np.int32
    )
    ds['ismip6shelfMelt_basin'].attrs = {
        'long_name': 'basin number for the MALI melt parameterisation',
        'convention': '1-based, 1-16, equal to ismip7BasinNumber + 1; '
                      '0 marks cells with no basin',
    }

    bfrn = ds_remapped['ismip7BFRNBin']
    ds['ismip7BFRNBin'] = bfrn.round().fillna(-1).astype(np.int32)
    ds['ismip7BFRNBin'].attrs = {
        'long_name': 'buttressing flux response number bin',
        'convention': '0-9; bin 0 is passive ice, bin 9 the most '
                      'buttressing-relevant; -1 marks cells with no bin',
    }

    ds['ismip7FloatingMask'] = (
        ds_remapped['ismip7FloatingMask'].round().fillna(0).astype(np.int32)
    )
    ds['ismip7FloatingMask'].attrs = {
        'long_name': 'ISMIP7 floating-ice mask',
        'convention': '1 where floating, 0 otherwise',
        'note': 'This is the observed ISMIP7 shelf extent, for diagnostics '
                'such as comparing modelled with observed shelf area. Melt '
                'aggregation should use the ice-sheet model own floating '
                'cells, since the calibration holds the model accountable '
                'for its own shelf extent.',
    }

    region = ds_remapped['ismip7ShelfRegion'].round().fillna(0)
    ds['ismip7ShelfRegion'] = region.astype(np.int32)
    ds['ismip7ShelfRegion'].attrs = {
        'long_name': 'ice-shelf region for calibration term J4',
        'convention': ', '.join(
            f'{value} = {name}' for name, value in REGION_CODES.items()
        ),
    }
    return ds


def cross_check_basins(ds_masks, region_mask_file, logger):
    """
    Compare the remapped basin field with MALI's existing region mask.

    The two are built from different sources -- ISMIP7 IMBIE2 v3 here, versus
    the ISMIP6 regions rasterised with ``geometric_features`` in 2022 -- so
    exact agreement is not expected.  What matters is that the *offset* is the
    expected one: an off-by-one would show near-zero agreement everywhere,
    whereas genuine boundary differences show up in individual basins.

    Agreement is reported in **both directions**, because the two masks do not
    cover the same cells and a one-directional figure is misleading.  A basin
    that ISMIP7 draws smaller than ISMIP6 scores high conditioned on ours and
    low conditioned on theirs; that is a real difference in the basin outlines,
    not an error.
    """
    if not os.path.exists(region_mask_file):
        logger.warning(f'region mask {region_mask_file} not found; skipping')
        return

    ds_region = xr.open_dataset(region_mask_file)
    if 'regionCellMasks' not in ds_region:
        logger.warning('no regionCellMasks in the region mask file; skipping')
        return

    masks = ds_region['regionCellMasks'].values
    # column index + 1 is MALI's 1-based basin number
    mali_basin = np.zeros(masks.shape[0], dtype=np.int32)
    for col in range(masks.shape[1]):
        mali_basin[masks[:, col] == 1] = col + 1

    ours = ds_masks['ismip6shelfMelt_basin'].values
    floating = ds_masks['ismip7FloatingMask'].values == 1
    both = (ours > 0) & (mali_basin > 0)
    if not both.any():
        logger.warning('no cells with both basin numbers; skipping check')
        return

    logger.info('')
    logger.info('cross-check against the existing MALI region mask:')
    for label, sel in (
        ('all cells', both),
        ('floating only', both & floating),
    ):
        if not sel.any():
            continue
        frac = 100.0 * (ours[sel] == mali_basin[sel]).mean()
        logger.info(f'  {label:14s} agreement {frac:5.1f}%  (n={sel.sum()})')

    if 100.0 * (ours[both] == mali_basin[both]).mean() < 80.0:
        logger.warning(
            '  LOW OVERALL AGREEMENT -- the basin numbering conventions are '
            'probably mismatched.  ISMIP7 is 0-based and MALI 1-based; see '
            'the module docstring.'
        )

    logger.info('  per basin, conditioned on each mask in turn:')
    logger.info(
        f'    {"basin":>5s} {"ours->theirs":>13s} {"n":>7s} '
        f'{"theirs->ours":>13s} {"n":>7s}'
    )
    for basin in range(1, masks.shape[1] + 1):
        sel_ours = both & (ours == basin)
        sel_theirs = both & (mali_basin == basin)
        if not (sel_ours.any() or sel_theirs.any()):
            continue
        a = (
            100.0 * (mali_basin[sel_ours] == basin).mean()
            if sel_ours.any() else float('nan')
        )
        b = (
            100.0 * (ours[sel_theirs] == basin).mean()
            if sel_theirs.any() else float('nan')
        )
        logger.info(
            f'    {basin:5d} {a:12.1f}% {int(sel_ours.sum()):7d} '
            f'{b:12.1f}% {int(sel_theirs.sum()):7d}'
        )
    logger.info(
        '  (a basin drawn smaller by ISMIP7 than by ISMIP6 scores high in the '
        'first column and low in the second; that is a real difference in the '
        'outlines, not an error)'
    )


def main(argv=None):
    args = parse_args(argv)

    mali_mesh_name = args.mali_mesh_name
    if mali_mesh_name is None:
        mali_mesh_name = os.path.splitext(
            os.path.basename(args.mali_mesh)
        )[0]

    with LoggingContext(__name__) as logger:
        logger.info(f'loading ISMIP7 masks from {args.ismip7_dir}')
        ds_masks = load_ismip7_masks(args.ismip7_dir, args.resolution)

        # any of the mask files defines the source grid; they share it
        src_grid_file = os.path.join(
            args.ismip7_dir, 'imbie2',
            f'basin_numbers_ismip{args.resolution}km_v2.nc',
        )

        ds_remapped = remap_to_mali(
            ds_masks, src_grid_file, args.mali_mesh, mali_mesh_name,
            args.resolution, args.method, args.ntasks, args.mapping_file,
            logger,
        )

        ds_out = to_integer_masks(ds_remapped)

        if args.region_mask is not None:
            cross_check_basins(ds_out, args.region_mask, logger)

        ds_out.attrs['source'] = (
            f'ISMIP7 masks from {args.ismip7_dir} remapped onto '
            f'{args.mali_mesh}'
        )
        ds_out.attrs['remap_method'] = args.method
        ds_out.attrs['history'] = (
            f'{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}: '
            f'{" ".join(sys.argv)}'
        )

        logger.info(f'writing {args.out_file}')
        ds_out.to_netcdf(args.out_file)

    return 0


if __name__ == '__main__':
    sys.exit(main())
