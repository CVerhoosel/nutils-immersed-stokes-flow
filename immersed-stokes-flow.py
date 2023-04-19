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

from nutils import cli, mesh, function, solver, testing
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

        # Construct the skeleton mesh: interfaces of the ambient domain that
        # border two non-empty elements in the trimmed domain
        skeleton_mesh = ambient_domain.interfaces.compress([all(domain.transforms.contains_with_tail(tr) for tr in neighbours)
            for neighbours in zip(ambient_domain.interfaces.transforms, ambient_domain.interfaces.opposites)])

        # Construct the ghost mesh: elements of the skeleton mesh that border
        # at least one trimmed element
        ghost_mesh = skeleton_mesh.compress([any(get_ref(ambient_domain, tr) != get_ref(domain, tr) for tr in neighbours)
            for neighbours in zip(skeleton_mesh.transforms, skeleton_mesh.opposites)])

        domain_porosity = domain.integrate(function.J(geom), degree=1)/numpy.prod(L)
        treelog.user(f'porosity: {domain_porosity:5.4f}')

        # Plot the meshes
        pp.meshes(domain, skeleton_mesh, ghost_mesh)

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

    return domain_porosity, sol, Qout

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

def get_ref(topo, trans):
    'return element in topo at the position of trans'
    index, tail = topo.transforms.index_with_tail(trans)
    return topo.references[index]

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
        porosity, sol, outflow = stokes_flow(L=(2.5,2.5), R=(0.4,0.7), mu=1.,
            beta=100., gammaskeleton=0.05, gammaghost=0.0005, pbar=1.,
            nelems=(12,12), degree=3, maxrefine=2, seed=123456, plotting='none')

        with self.subTest('porosity'):
            self.assertAlmostEqual(porosity, 0.65403913789, places=10)

        with self.subTest('outflow'):
            self.assertAlmostEqual(outflow, 0.01748244953, places=10)

        with self.subTest('solution'):
            self.assertAlmostEqual64(sol['u'], '''
                eNoN0Y9PVVUcAPByScp0hRuKzg3iAu/ee875fh8KypBaWy0UlAxxiErONIM5NqWGlZCVbeAPUGEpLTZk
                yOavB6KCG+ic6MzBffeec+59970HvFdiG8X8FWmkY5n/wudz2vzDzJMnUvOglzyDN51LUO2cBl9SJxSQ
                53CTToLq2UCzA5aVS3/zr+DlxvcZN8zj3jTJtCJodBbgCacbzjsH4c7oKcgnsTiLLsRXqZdkE5W3iG3W
                Wj6trxF3oUoc4s1e1yr1Ntht5Ah0ycdwU+6HKTsVkkNlcIv8Cq+xeGxlDo2lB8RZweUymast0WqDKep3
                7n532H6P1LAv5BSsFh3QIVrYAbmFRtwiFqG1EAev40Goh5VQTCgtTmvQK4a3jHyT+tPwMe3P8IfsiRyD
                EqHgC06gUAyQzfakNjO0h8yhD1gpvIVXgWExpEMT7SefhSuVhOT20cfKDLbX0wlrnFL8R8RhnfDoV91V
                8it4xb6o/ygnxGx9TE3APGjGIFuAW8nX7GGwIfpR9EHS4kgri0/zYGGgBTPtFfCX3c1VOGSuS99lfc7O
                m/+aP/PnYgP8TRw8QyuwSL/HKsO+hdeTliduUnpYZehdfF89iSHtChtXd1jHnRdGHR83RjDL8pEou+BG
                scrdh1mhp2ye6kvcGemNTI/sYLeDT2E76cIB1gbLGeGjvN5/zs/9n9Ayf+WSaSsHLeiWE6jIZVjntLNy
                NSWar2SkbE19mzwK9jGNlWEW9mMVbiK79HhxhC8Wc7X7vBFrSFNgHCt4O/r4fLgg+8hY2lFl+0hh+I3Q
                M8fQPg4U6DHQCAMYwBw8g9nsMBsiPeESli8/xUw+iTN4NcwS3e4HzjGn3c0aTvZcIzGsgG4UP1iDZoz4
                nX/58m83/gdDkAP5sDL0EI6KVpR8HNaKVcKCektJ38jmwk49xnGN6sEUf49hm7nWorA3HIslbA+W02o8
                GWx6ab4a18mI3uyGzLPeBP/6pfeMw5kzh/KMeZYipmSvNh8trQsD2rfYFryB150i3Bvos58E4iwDJgb7
                l142O9PRPmVn4D67A3/RazAxeBk9zja8S95xFrE7Zq35P+k4iP8=''')
            self.assertAlmostEqual64(sol['p'], '''
                eNoNjj1I1XEUhilEo6XBMLAMRCpJ+H/9/r/nTUTMMMvM1MrIr3KwLXFoSLgqLSG06qA0CKFLgwg5BHWD
                IoKGvJgUFNwoIiINRIOiuEpnOi/neZ/DeUet+tSvizqnC+rRgFr0i+uEvMepU5dsf1pnLXWpUQXuUU0l
                I+SpUoPOqNlos80mOZXoDXMs8ZxPlKjaeJv5HdZIdVh7tM1fvvGRVX6wz3bt6tUVu1pl5j+KVarP5HjG
                a35yQHVqNbPG2Bpf2GK3HvGQWeZ5ygd+s1dF2rT0kiyvzBviFhnGuM1NehlmgimmmbSP73KEeq7SzkFe
                +DZf5gd93qfcsGYntRQZb2A/sz6XTrvVZNF9T5f8MTqAMuOniMj6jXTezcUPwifxY7fLL/uTXOY8y9yn
                guN+0w0mI1EQjkVDyY6r8zGLfKVROX80HU3K40zUHV2LClEmOZGu+X5l3SF3J14PC8F40Bo0hV1x4v64
                ySSOZ4KJt+Ur9SsDNfloIfkP0RKMQw==''')
