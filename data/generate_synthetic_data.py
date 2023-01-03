#! /usr/bin/env python3
#
# This script implements the `generate_synthetic_data` function.

from nutils import cli, mesh, function, export
import numpy, numpy.random, itertools, pathlib, json, typing, treelog
from matplotlib import collections, colors, patches, colormaps, cm

def generate_synthetic_data(ndims:int, R:typing.Tuple[float,float], L:float, nvox:int, nint:int, dtype:str, seed:int):

    '''
    Generate synthetic data.

    .. arguments::

        ndims [2]
            Number of dimensions.

        R [0.4,0.7]
            Range of inclusion radii.

        L [2.5]
            Domain size.

        nvox [50]
            Number of voxels per direction.

        nint [8]
            Number for integration points.

        dtype [<i2]
            Gray scale data type.

        seed [123456]
            Seed for the random number generator for the radii.

    .. presets::

        3D
            ndims=3
            nvox=25
    '''

    # Construct the voxel topology
    topo, geom = mesh.rectilinear([numpy.linspace(0,L,nvox+1)]*ndims)

    # Construct the levelset function
    rng = numpy.random.default_rng(seed)
    r   = rng.uniform(*R)
    lvl = r-function.norm2(geom-L/2)
    treelog.user(f'Inclusion at {L/2,L/2} with radius {r:3.2f}.')
    for xvert in itertools.product([0,L],repeat=ndims):
        r   = rng.uniform(*R)
        lvl = function.max(r-function.norm2(geom-xvert), lvl)
        treelog.user(f'Inclusion at {xvert} with radius {r:3.2f}.')

    # Compute and save the volume fractions
    data = topo.elem_mean(function.heaviside(lvl), geometry=geom, ischeme=f'uniform{nint}')

    # Rescale the data to the dtype range
    nf   = numpy.iinfo(dtype)
    data = (nf.max-nf.min)*data+nf.min

    # Save the raw data
    fraw = pathlib.Path(__file__).resolve().parent / f'synthetic_{"x".join(str(d) for d in [nvox]*ndims)}.raw'
    data.astype(dtype).tofile(str(fraw))

    # Write the json file
    with fraw.with_suffix('.json').open('w') as fout:
        json.dump({'fname'  : str(fraw.name),
                   'shape'  : [nvox]*ndims  ,
                   'dtype'  : dtype         ,
                   'order'  : 'C'           ,
                   'spacing': [L/nvox]*ndims }, fout)

    # Plot the data (if 2D)
    if ndims==2:
        with export.mplfigure('voxeldata.png') as fig:
            ax = fig.add_subplot(111, aspect='equal', title='voxel data')
            pixels = [patches.Rectangle((L/nvox)*numpy.array(ij), (L/nvox), (L/nvox)) for ij in itertools.product(range(nvox), repeat=2)]
            norm = colors.Normalize(vmin=numpy.min(data), vmax=numpy.max(data))
            cmap = colormaps['binary']
            ax.add_collection(collections.PatchCollection(pixels, facecolors=cmap(norm(data)), edgecolors=(0,0,0,0.25), linewidths=0.5))
            ax.autoscale(enable=True, axis='both', tight=True)
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

if __name__ == '__main__':
    cli.run(generate_synthetic_data)