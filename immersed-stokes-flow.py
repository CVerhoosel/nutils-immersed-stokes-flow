#! /usr/bin/env python3
#
# Authors: Clemens Verhoosel, Sai Divi
#
# This example demonstrates the skeleton-stabilized immersogeometric simulation
# of a Stokes flow through a porous medium. Details can be found in
# [this review] (https://doi.org/10.48550/arXiv.2208.14994). The porous medium
# domain, Ω, is constructed based on an analytical level set function.
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
from nutils.expression_v2 import Namespace
from typing import Tuple
import numpy, treelog, numpy.linalg, itertools
import postprocessing

def stokes_flow(L:Tuple[float,...], R:Tuple[float,float], mu:float, beta:float, gammaskeleton:float, gammaghost:float, pbar:float, nelems:Tuple[int,...], degree:int, maxrefine:int, seed:int, plotting:str):

    '''
    Stokes flow through a porous medium obtained from scan data.

    .. arguments::

        L [2.5,2.5]
            Domain size.

        R [0.4,0.7]
            Range of inclusion radii.

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

        seed [123456]
            Seed for the random number generator for the radii.

        plotting [matplotlib]
            Plotting format (matplotlib, vtk or none)

    .. presets::

        3D
            L={2.5,2.5,2.5}
            nelems={12,12,12}
            maxrefine=1
            plotting=vtk
    '''

    # Pre-processing steps to convert the scan data into a trimmed mesh
    with treelog.context('pre-processing'):

        # Construct the ambient domain mesh
        assert len(nelems)==len(L)
        ambient_domain, geom = mesh.rectilinear([numpy.linspace(0,l,ne+1) for ne,l in zip(nelems,L)])

        # Load the post-processor
        pp = postprocessing.initialize(plotting, len(L), degree, maxrefine, L, geom)

        # Determine the levelset function
        levelset = get_levelset(L, R, geom, seed)

        # Trim the domain
        domain = ambient_domain.trim(levelset, maxrefine=maxrefine)
        background_mesh, skeleton_mesh, ghost_mesh = construct_meshes(ambient_domain, domain)

        domain_porosity = domain.integrate(function.J(geom), ischeme='uniform1')/numpy.prod(L)
        treelog.user(f'porosity: {domain_porosity:5.4f}')

        # Plot the meshes
        pp.meshes(domain, background_mesh, skeleton_mesh, ghost_mesh)

    # Solving the Stokes problem using immersed isogeometric analysis
    with treelog.context('immersed iga solver'):

        # Namespace initialization
        ns = Namespace()
        ns.δ = function.eye(domain.ndims)
        ns.x = geom
        ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))

        ns.μ    = mu
        ns.h    = numpy.linalg.norm(numpy.array(L)/numpy.array(nelems))
        ns.β    = beta
        ns.γs   = gammaskeleton
        ns.γg   = gammaghost
        ns.pbar = pbar

        # Construct the velocity-pressure basis
        ns.ubasis = domain.basis('spline', degree=degree).vector(domain.ndims)
        ns.pbasis = domain.basis('spline', degree=degree)

        ns.dnΔubasis = dnΔ(ns.ubasis, ns.x, 'n_i h' @ ns, count=degree)
        ns.dnΔpbasis = dnΔ(ns.pbasis, ns.x, 'n_i h' @ ns, count=degree)

        # Velocity and pressure fields
        ns.u = function.dotarg('u', ns.ubasis)
        ns.p = function.dotarg('p', ns.pbasis)
        ns.σ_ij = 'μ (∇_j(u_i) + ∇_i(u_j)) - δ_ij p'

        ns.dnΔu = dnΔ(ns.u, ns.x, 'n_i h' @ ns, count=degree)
        ns.dnΔp = dnΔ(ns.p, ns.x, 'n_i h' @ ns, count=degree)

        # Residual volume terms
        resu = domain.integral('∇_j(ubasis_ni) σ_ij dV' @ ns, degree=2*degree)
        resp = domain.integral('-∇_k(u_k) pbasis_n dV' @ ns, degree=2*degree)

        # Dirichlet boundary terms
        dirichlet_boundary = domain.boundary['trimmed,top,bottom' if domain.ndims==2 else 'trimmed,top,bottom,front,back']
        resu += dirichlet_boundary.integral('-σ_ij n_i ubasis_nj dS' @ ns, degree=2*degree)
        resu += dirichlet_boundary.integral('-μ ((∇_j(ubasis_ni) + ∇_i(ubasis_nj)) n_j - (β / h) ubasis_ni) u_i dS' @ ns, degree=2*degree)
        resp += dirichlet_boundary.integral('pbasis_n u_i n_i dS' @ ns, degree=2*degree)

        # Inflow boundary term
        resu += domain.boundary['left'].integral('pbar n_i ubasis_ni dS' @ ns, degree=2*degree)

        # Skeleton stabilization term
        resp += skeleton_mesh.integral(f'-γs h dnΔpbasis_n dnΔp dS' @ ns, degree=2*degree)

        # Ghost stabilization term
        resu += ghost_mesh.integral('(γg / h) dnΔubasis_ni dnΔu_i dS' @ ns, degree=2*degree)

        # Solve the linear system
        sol = solver.solve_linear(('u', 'p'), (resu, resp))

    # Post-processing of results
    with treelog.context('post-processing'):
        pp.solution(domain, ns, sol)

    # Compute the averaged in- and outflow
    Ain , Qin , pin  = domain.boundary['left'] .integrate(['dS', '-u_i n_i dS', 'p dS']@ns, degree=degree, arguments=sol)
    Aout, Qout, pout = domain.boundary['right'].integrate(['dS', 'u_i n_i dS' , 'p dS']@ns, degree=degree+1, arguments=sol)
    treelog.user(f'(in/out)flow area    : {Ain} / {Aout}')
    treelog.user(f'(in/out)flow pressure: {pin/Ain} / {pout/Aout}')
    treelog.user(f'(in/out)flow flux    : {Qin} / {Qout}')
    treelog.user(f'(in/out)flow velocity: {Qin/Ain} / {Qout/Aout}')

    return domain_porosity, numpy.concatenate([sol['u'],sol['p']]), Qout

def get_levelset(L, R, geom, seed):

    # Construct the levelset function
    rng      = numpy.random.default_rng(seed)
    r        = rng.uniform(*R)
    levelset = function.norm2(geom-numpy.array(L)/2)-r
    treelog.user(f'Inclusion at {numpy.array(L)/2} with radius {r:3.2f}.')
    for xvert in itertools.product(*([0,l] for l in L)):
        r        = rng.uniform(*R)
        levelset = function.min(function.norm2(geom-xvert)-r, levelset)
        treelog.user(f'Inclusion at {xvert} with radius {r:3.2f}.')

    return levelset

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

def dnΔ(f, x, n, count=1):
    'construct the count-times normal gradient of the jump of f'
    f = function.jump(f)
    for i in range(count):
        f = function.grad(f, x)
    for i in range(count):
        f @= n
    return f

if __name__ == '__main__':
    cli.run(stokes_flow)

# Unit testing
class test(testing.TestCase):

  def test_2D(self):
    porosity, lhs, outflow = stokes_flow(L=(2.5,2.5), R=(0.4,0.7), mu=1., beta=100.,
                                         gammaskeleton=0.05, gammaghost=0.0005, pbar=1.,
                                         nelems=(12,12), degree=3, maxrefine=2, seed=123456, plotting='none')

    # Porosity
    with self.subTest('porosity'): self.assertAlmostEqual(porosity, 0.6540391378932529, places=10)

    # Flux
    with self.subTest('outflow'): self.assertAlmostEqual(outflow, 0.01748244952959246, places=10)

    # Solution vector
    with self.subTest('lhs'):
        self.assertAlmostEqual64(lhs,
         '''eNoNkntQVHUUx9NRNCbHtEQly8fy2N17f79zd+/e80PR1DJREETEFBEfiaZjTIaNWoBa1uD7RSmOzqAB
            M2qCkGiNSI7omOnufe8LXPJRQ5JPfJAOafev853zOd8zZ845R+S/5VR9X3wqPc09o6+bJ2mxeYTWjKil
            GdxzeoHvpPbEOfwYv6Kk8Nd9yepy79ee8/JeIUEnjmy6xxwM+8x6etzcQi9fq6JpXDT05YdCD17gxnB2
            9aC2WMlUu53p2g26WtuqlgsBJU/YYRzmdtIT+gN6Qd9Iu4x4Oiq0jF7k/qC9yCCoICYfzW/WjmmqjnqK
            w+0oDcbZvwpsDLQY73Ml5HO9i07Tqmm1dpBs1hfwkUA2ifCldADtA1vodjqFzuZ4fnbCDmdBy4LWdfH7
            W3Y7boenk8f6TZqj2eClytEsrZmbZ3Q6eofWcK/xd0keHQlnKYHZ1EXL+Ebu43ChbcioymsPbD1JUWIt
            TTfz4Kk2ADZpic6zgan6F/QV4yfn93qH9qrzpn0IpNJyCJLBsIj7ktwL7mib0XZ3xLBIBRmUkAhZ/oMg
            Gcn0oVGv2ulWeaZrpbKKHJf/lQ+oz7U59BFnwlG+ALKdt0hhuGbouRFseK7tFCkMTYBJ9kMQcvxM2u1L
            lb3mS+8mtd3bCklKDddG6gJtsDqwHpJCT8hAe83wFZHTke7WpeRS8AnN505AMzlMGeHUa+p2348+1fcR
            v8xX6O5WxoJC6/UOsOkIm8xKstwe15Zm88Qtih/H3Q+eIQ6yDJKgEVZDLrfSOUjbqQ7T+jnuqHughCvz
            t0OBWgk1agyt089wNxN22fJbs8L9Q89Mr2O+P8MZRffQZvDDWDgKY8g2cpU7Fc4hafoSkNRO6KkW075a
            fWCyudusDCS1jEps4qJIBj9X+0a5Ikdpf6prrft9Bi/oVTqWptEpoXt0l1YButpOM7WpmkK3KzbXXNKP
            rnBGmQFv8ZU43ymvIacosWEhHA05ZA0s54vhULDM2vk0mKlHnOWBkHxMGOKbJd7ybpN6X031DlRsWpd+
            2hEDiuME+B0b4HDwPJwzs6HIf8Z47B+geGnHlUaxQa51gVFleGC9UQ2/OUtgeLABEs3FcIN714wll+VS
            2Y+j2TyWx7JYKstgc9lCNpU9wgUIGECRzWAzrfwHbIqlZrGJrBu3oB1H4lqMoI2NZylsskUnW3ESE1kf
            5sMqbMDz2IZ9mN3i6ZY/06rwsHdYX/YfPsO/sAUNvI39rdx0lss+tLraLOdzjGJvsOuo4K/4O97BwSyZ
            pVlOzmIdeAM7sSc7icewAqvxLIbxKUazXuyhpS5iE16yfAW4CouwBNfgJ5iLn2IpfoflWGZN/C3G4zic
            g9PxLWyW0qUYKV+KSB5cYlXOwNHYy+Lj8U2skBRPuWi468R2T4OUiJmIGGPx91DAJum+p1qscv0Aja5f
            xB6SLE3AbJyGMh7At9EpPRTz3WsFCiVCgfuFmCy5sA5v4USmSAmeYnesq0jIEeYL3UKRO8nTIeWxJnGY
            uMH1D3TTddafTIJZLrfYJZa5Xa79tFSP1cZpC7mIUOv+H2QqFVA=''')
