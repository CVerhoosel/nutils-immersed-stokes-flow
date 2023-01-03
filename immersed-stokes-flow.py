#! /usr/bin/env python3
#
# Authors: Clemens Verhoosel, Sai Divi
#
# This example demonstrates the skeleton-stabilized immersogeometric simulation
# of a Stokes flow through a porous medium. Details can be found in
# [this review] (https://doi.org/10.48550/arXiv.2208.14994). The porous medium
# domain, Ω, is constructed from voxel data, either created artificially using
# the `generate_synthetic_data.py` script in the `data` directory, or obtained
# directly from scan data.
#
# In the considered formulation, the (no slip) essential boundary conditions
# are imposed weakly using Nitsche's method. A pressure drop across the porous
# medium is imposed using Neumann conditions:
#
#   - ∇·(2μ ∇^s u) + ∇ p = 0    in Ω
#                 ∇ . u  = 0    in Ω
#                     u  = 0    on Γ_D
#     2μ ∇^s u · n - p n = 1    on Γ_N
#
# The problem is discretized using equal-degree spline spaces for the velocity
# and pressure spaces. Stability of the discretization is ensured through the
# application of a pressure-stabilization term acting on the skeleton of the
# background mesh, and a ghost-stabilization term for the velocity acting on
# the ghost mesh.

from nutils import cli, mesh, function, topology, solver, elementseq, transformseq, testing
import json, numpy, treelog, pathlib, numpy.linalg, typing
import postprocessing

def stokes_flow(name:str, threshold:float, mu:float, beta:float, gammaskeleton:float, gammaghost:float, pbar:float, nelems:typing.Tuple[int,...], degree:int, maxrefine:int, plotting:str):

    '''
    Stokes flow through a porous medium obtained from scan data.

    .. arguments::

        name [synthetic_50x50]
            File name of scan-data.

        threshold [0.]
            Grayscale segmentation threshold.

        mu [1.]
            Fluid viscosity.

        beta [100.]
            Nitsche parameter.

        gammaskeleton [0.05]
            Skeleton parameter.

        gammaghost [0.0005]
            Ghost parameter.

        pbar [1.]
            Pressure drop

        nelems [12,12]
            Number of elements.

        degree [3]
            Polynomial degree.

        maxrefine [2]
            Bisectioning steps.

        plotting [matplotlib]
            Plotting format (matplotlib, vtk or none)

    .. presets::

        3D
            name=synthetic_25x25x25
            nelems={12,12,12}
            maxrefine=1
            plotting=vtk
    '''

    # Pre-processing steps to convert the scan data into a trimmed mesh
    with treelog.context('pre-processing'):

        # Read the voxel data
        lengths, data = read_voxel_data(name)

        treelog.user(f'threshold (scan): {threshold}')
        treelog.user(f'porosity  (scan): {(data<threshold).sum()/data.size:5.4f}')

        # Construct the ambient domain mesh
        assert len(nelems)==len(lengths)
        ambient_domain, geom = mesh.rectilinear([numpy.linspace(0,l,ne+1) for ne,l in zip(nelems,lengths)])

        # Load the post-processor
        pp = postprocessing.initialize(plotting, data.ndim, degree, maxrefine, lengths, geom)

        # Determine the levelset function
        levelset_domain, levelset = get_levelset(lengths, data, ambient_domain, geom, degree, pp)

        # Recalibrate the threshold
        intensities = levelset_domain.sample('uniform', 2**(maxrefine+1)).eval(levelset)
        levelset_threshold = numpy.sort(intensities)[(intensities.size*(data<threshold).sum())//data.size]

        treelog.user(f'threshold (lvl.): {levelset_threshold}')
        treelog.user(f'porosity  (lvl.): {(intensities<levelset_threshold).sum()/intensities.size:5.4f}')

        # Trim the domain
        domain = ambient_domain.trim(levelset_threshold-levelset, maxrefine=maxrefine, leveltopo=levelset_domain)
        background_mesh, skeleton_mesh, ghost_mesh = construct_meshes(ambient_domain, domain)

        domain_porosity = domain.integrate(function.J(geom), ischeme='uniform1')/numpy.prod(lengths)
        treelog.user(f'porosity  (trim): {domain_porosity:5.4f}')

        # Plot the meshes
        pp.meshes(domain, background_mesh, skeleton_mesh, ghost_mesh)

    # Solving the Stokes problem using immersed isogeometric analysis
    with treelog.context('immersed iga solver'):

        # Namespace initialization
        ns = function.Namespace()
        ns.x    = geom
        ns.μ    = mu
        ns.h    = numpy.linalg.norm(numpy.array(lengths)/numpy.array(nelems))
        ns.β    = beta
        ns.γs   = gammaskeleton
        ns.γg   = gammaghost
        ns.pbar = pbar

        # Construct the velocity-pressure basis
        ns.ubasis = domain.basis('spline', degree=degree).vector(domain.ndims)
        ns.pbasis = domain.basis('spline', degree=degree)

        ns.Δubasis = function.jump(ns.ubasis)
        ns.Δpbasis = function.jump(ns.pbasis)

        # Velocity and pressure fields
        ns.u_i = 'ubasis_ni ?lhsu_n'
        ns.p   = 'pbasis_n ?lhsp_n'

        ns.Δu_i = 'Δubasis_ni ?lhsu_n'
        ns.Δp   = 'Δpbasis_n ?lhsp_n'

        # Residual volume terms
        resu = domain.integral('(μ ubasis_ni,j (u_i,j + u_j,i) - ubasis_nk,k p) d:x'@ns, degree=2*degree)
        resp = domain.integral('-u_k,k pbasis_n d:x'@ns, degree=2*degree)

        # Dirichlet boundary terms
        dirichlet_boundary = domain.boundary['trimmed,top,bottom' if domain.ndims==2 else 'trimmed,top,bottom,front,back']
        resu += dirichlet_boundary.integral('(-μ ((u_i,j + u_j,i) n_i ubasis_nj + (ubasis_ni,j + ubasis_nj,i) n_i u_j) + μ (β / h) ubasis_ni u_i + p ubasis_ni n_i) d:x'@ns, degree=2*degree)
        resp += dirichlet_boundary.integral('pbasis_n u_i n_i d:x'@ns, degree=2*degree)

        # Inflow boundary term
        resu += domain.boundary['left'].integral('pbar n_i ubasis_ni d:x'@ns, degree=2*degree)

        # Skeleton stabilization term
        from string import ascii_lowercase as alphabet
        dn = lambda function, start, stop: f'{function},{alphabet[start:stop]} {" ".join([f"n_{index}" for index in alphabet[start:stop]])}'
        resp += skeleton_mesh.integral((f'-γs h^{2*degree+1} {dn("Δpbasis_n", 0, degree)} {dn("Δp_", degree, 2*degree)} d:x')@ns, degree=2*degree)

        # Ghost stabilization term
        resu += ghost_mesh.integral((f'γg h^{2*degree-1} {dn("Δubasis_ni", 0, degree)} {dn("Δu_i", degree, 2*degree)} d:x')@ns, degree=2*degree)

        # Solve the linear system
        sol = solver.solve_linear(('lhsu', 'lhsp'), (resu, resp))

    # Post-processing of results
    with treelog.context('post-processing'):
        pp.solution(domain, ns, sol)

    # Compute the averaged in- and outflow
    Ain , Qin , pin  = domain.boundary['left'] .integrate(['d:x', '-u_i n_i d:x', 'p d:x']@ns, degree=degree, arguments=sol)
    Aout, Qout, pout = domain.boundary['right'].integrate(['d:x', 'u_i n_i d:x' , 'p d:x']@ns, degree=degree+1, arguments=sol)
    treelog.user(f'(in/out)flow area    : {Ain} / {Aout}')
    treelog.user(f'(in/out)flow pressure: {pin/Ain} / {pout/Aout}')
    treelog.user(f'(in/out)flow flux    : {Qin} / {Qout}')
    treelog.user(f'(in/out)flow velocity: {Qin/Ain} / {Qout/Aout}')

    return domain_porosity, numpy.concatenate([sol['lhsu'],sol['lhsp']]), Qout

# Function to read the voxel data
def read_voxel_data(name):

    # Json file name
    fname = pathlib.Path(__file__).resolve().parent / 'data' / f'{name}.json'
    assert fname.is_file(), f'"{fname.name}" not found in data directory. Synthetic data can be generated using the "generate_synthetic_data" script in the data directory.'

    # Read the Json file
    with fname.open() as jsonfile:
        jsondict = json.load(jsonfile)
    shape   = jsondict['shape']
    spacing = jsondict['spacing']
    lengths = [sh*sp for sh, sp in zip(shape, spacing)]

    # Raw file name
    rawfile = fname.with_name(pathlib.Path(jsondict['fname']).name)
    assert rawfile.is_file(), f'File "{rawfile.name}" does not exist'

    # Read the raw data file
    with rawfile.open('rb') as datafile:
        data = numpy.fromfile(file=datafile, dtype=jsondict['dtype'], count=numpy.prod(shape)).reshape(shape, order=jsondict['order'])
    rng = (numpy.min(data), numpy.max(data))

    treelog.user(f'domain size     : {"×".join(str(d) for d in lengths)}')
    treelog.user(f'voxel data shape: {"×".join(str(d) for d in shape)}')
    treelog.user(f'voxel spacing   : {"×".join(str(d) for d in spacing)}')
    treelog.user(f'intensity range : {rng[0]}, {rng[1]}')

    return lengths, data

def get_levelset(lengths, data, ambient_domain, geom, degree, pp):

    # Construct the levelset domain by refining the ambient domain until
    # (approximately) matching the voxel grid.
    nref = min(numpy.floor(numpy.log2(sh/ne)).astype(int) for sh, ne in zip(data.shape,ambient_domain.shape))
    levelset_domain  = ambient_domain.refine(nref)
    levelset_spacing = numpy.array([l/sh for sh, l in zip(levelset_domain.shape, lengths)])

    treelog.user(f'levelset shape  : {"×".join(str(d) for d in levelset_domain.shape)}')
    treelog.user(f'levelset spacing: {"×".join(str(d) for d in levelset_spacing)}')

    # Sample the levelset domain
    levelset_sample = levelset_domain.sample('uniform', 2)
    levelset_points = levelset_sample.eval(geom)

    # Find the voxel data values corresponding to the levelset points
    indf = levelset_points/levelset_spacing
    indi = numpy.maximum(numpy.minimum(numpy.floor(indf),numpy.array(data.shape)-1),0).astype(int)
    levelset_data = data[tuple(indi.T)]

    # Construct the voxel intensity function
    intensity = levelset_sample.basis().dot(levelset_data)

    # Smoothen the intensity data using a B-spline basis
    basis    = levelset_domain.basis('spline', degree)
    den, num = levelset_sample.integrate([basis, intensity*basis])
    levelset = basis.dot(num/den)

    # Plot the voxel data and levelset function
    pp.data(data)
    pp.levelset(levelset_domain, levelset)

    return levelset_domain, levelset

def construct_meshes(ambient_domain, domain):

    # Extract the background mesh
    background_mesh = topology.SubsetTopology(ambient_domain, [ref  if domain.transforms.contains_with_tail(tr) else  ref.empty for tr, ref in zip(ambient_domain.transforms,ambient_domain.references)])

    # Get the skeleton mesh
    skeleton_mesh = background_mesh.interfaces

    # Get the ghost mesh
    ghost_references = []
    ghost_transforms = []
    ghost_opposites  = []
    for skeleton_tr, skeleton_ref, skeleton_opp in zip(skeleton_mesh.transforms, skeleton_mesh.references, skeleton_mesh.opposites):
      for tr in skeleton_tr, skeleton_opp:

        # Find the corresponding element in the background mesh
        background_index = background_mesh.transforms.index_with_tail(tr)[0]

        # Find the corresponding element in the trimmed mesh
        index = domain.transforms.index_with_tail(tr)[0]

        # Mark as a ghost interface if the corresponding element was trimmed
        if background_mesh.references[background_index]!=domain.references[index]:
            assert background_mesh.transforms[background_index]==domain.transforms[index]
            assert background_mesh.opposites[background_index] ==domain.opposites[index]
            ghost_references.append(skeleton_ref)
            ghost_transforms.append(skeleton_tr)
            ghost_opposites.append(skeleton_opp)
            break

    ghost_references = elementseq.References.from_iter(ghost_references, skeleton_mesh.ndims)
    ghost_opposites  = transformseq.PlainTransforms(ghost_opposites, todims=background_mesh.ndims, fromdims=skeleton_mesh.ndims)
    ghost_transforms = transformseq.PlainTransforms(ghost_transforms, todims=background_mesh.ndims, fromdims=skeleton_mesh.ndims)
    ghost_mesh = topology.TransformChainsTopology('X', ghost_references, ghost_transforms, ghost_opposites)

    return background_mesh, skeleton_mesh, ghost_mesh

if __name__ == '__main__':
    cli.run(stokes_flow)

# Unit testing
class test(testing.TestCase):

  def test_synthetic_50x50(self):
    porosity, lhs, outflow = stokes_flow(name='synthetic_50x50', threshold=0., mu=1., beta=100.,
                                         gammaskeleton=0.05, gammaghost=0.0005, pbar=1.,
                                         nelems=(12,12), degree=3, maxrefine=2, plotting='none')

    # Porosity
    with self.subTest('porosity'): self.assertAlmostEqual(porosity, 0.6520314578104903, places=10)

    # Flux
    with self.subTest('outflow'): self.assertAlmostEqual(outflow, 0.01617581173359397, places=10)

    # Solution vector
    with self.subTest('lhs'):
        self.assertAlmostEqual64(lhs,
        '''eNodUgtQVFUYtpHEDEMLC/ARD9mFe8/5z112739SfGCJINqqaYwUoek4Kg4+GB8ZOmkCJlkqrihqkcoo
           osUMZlpDjZmCwu7e175YQcb3EyfzkaM1dqf5Zs78833zfd/855xSb4VyQBvma6LRRIXVvlqI8q2EOv9y
           +DqtGg6Lp2GLeJU6fV9qleJv3mt0nrvCscdbLi3TjouttNb3GOp9DZDhK4DYwGy4n7oXlos9kCseob+I
           27UOY6D6lzpJVZUe1qrGqGFpj1InTdejqBWijBvQqJfBViMK8vwvwShhNWSRdnhK7tIb5LJeaBC9QI8M
           NAfaQhWWrMCc1Lv6Y7KIFBshKNK+h0NaN3XqVWSd72eySHxEC2kdREAN3KLvkffJ29bBgtdSYhmRNKuz
           ylIXPEKGGPuhWIthEVoRSNp4Uq2npWn+1cJT4Tb5kTbBHBjEkqAYFpJ80hwfdSHuQnHijqRqstm6AEb4
           bOyRlsTmarvJJKOP0Ul7+drILuO8/mogzr8SbtON7CqVWK1YQz8M7U1+PTmlK7OrhA6yRrDJ/g2st7EL
           Io1j2m46Tdkq2dSB0holQbmpGOpSMUY4weKpi/0qDICBnZld33RFDn+WvI9u6mBsSbiS5QX7Qu+UW8o2
           YaanAha0l8mN7u3e8d6PtSX6ofBx1pLWzJTUEVCf+jyx4ZIn/CRcSneE+rBTQg3rTz+HxSRJ3aw5PSc9
           je4m9zl3vH2O1yVl+DRrO4s0jrAC/1B4Zp3YLSZ9a1llCYkTA9HwAZnPzoCLRbB/jC1Go5KnNCkuZafS
           KR1VS1kRxOk3WYYWyyRjMW1M3ZBQ2VkbdAXvd/QIAeEiKYd+LMhusFi2lC0TcsRHwRSrJfiuUEm7tUY2
           Tv2MrVfnU5seKUz1K12nUg6K58UWI0O/ooK6TJ+sP4CHUM3WsTLwgI/8Yf6lGfq/EFDXs61qP3pWz9da
           oVCfQsdbugLltixbjaJ5jnuuu1XvPW9tSmlQZI/Nt5hNm2GxEIImYwIbrn8BJfpgrQGKvCdtE7zR6bPo
           Rb2qrakt1/uREqdXByLZVfET9rd4CRYIb7AlvrWsw6inbqOvKkC+p8g2pd0p9/fesZ3Tiv3J7CdjDdPI
           U3giJLIDxiq21tcXviMVil0nnv1aGzp4AZ/N8/hkE3m8kE/nQ/jvmIBuBD7NZKbwbJ7DnXwGn8QT+QWs
           w5dxIfoxgY81+RxTzf7/HMst/AV+HQ/iCQxjH27lo3kun2oi19Qk0xvPY3k3hkzvHXyF28y8fD7TbEU+
           lA/gcXw4v2JqZ9GL9zCWjzEbZ/BM/iZ/jvexF4/mp83kBvwBz+B1jOZ2PpIP4vdQNR2KmVpuogIr8Svc
           hi68hpexEXea8xbchBtwDDpxroks7IcN8lB5nHxajsVsnIZvmbtGI8FclPGanCNzx4P0AvsuR4l8V47B
           h7Iqi6Y705xmytmOFek2tk8aaT/kWCl3mmiRj5kdo3GjvNbxor1eWp8GELQdtgccdXIvHIZl5m0KeNQR
           ac+y3WEL2U7mksam/2mfJzuxBVH+1D7K9o7Jh9kK1gQuSbQTuRiT0yvZayQZCLtCVxhV0Jre4/gP50we
           yg==''')