# Post-processing class accompanying `immersed-stokes-flow.py`

from nutils import export
import numpy
from matplotlib import collections, colors

# Initialize the post-processor
def initialize(plotting, ndims, *args):

    if plotting.lower()=='matplotlib' and ndims==2:
        return MatplotlibPlotter(*args)
    elif plotting.lower()=='vtk':
        return VTKWriter(*args)

    return PostProcessor(*args)

# Base class (no plotting)
class PostProcessor:

    def __init__(self, degree, maxrefine, lengths, geom):
        self.degree    = degree
        self.maxrefine = maxrefine
        self.lengths   = lengths
        self.geom      = geom

    def meshes(self, domain, skeleton_mesh, ghost_mesh):
        pass

    def solution(self, domain, ns, sol):
        pass

# Matplotlib plotting for 2D cases
class MatplotlibPlotter(PostProcessor):

    def meshes(self, domain, skeleton_mesh, ghost_mesh):

        bezier = domain.sample('bezier', 2**self.maxrefine+1)
        points = bezier.eval(self.geom)
        with export.mplfigure('immersed_domain.png') as fig:
            ax = fig.add_subplot(111, aspect='equal', title='domain')
            ax.autoscale(enable=True, axis='both', tight=True)
            im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, numpy.zeros(points.shape[0]), shading='gouraud', cmap='binary')
            fig.colorbar(im).remove()

        sbezier = skeleton_mesh.sample('bezier', 2)
        spoints = sbezier.eval(self.geom).reshape(-1,2,domain.ndims)
        with export.mplfigure('skeleton_mesh.png') as fig:
            ax = fig.add_subplot(111, aspect='equal', title='skeleton mesh')
            ax.autoscale(enable=True, axis='both', tight=True)
            im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, numpy.zeros(points.shape[0]), shading='gouraud', cmap='binary')
            ax.add_collection(collections.LineCollection(spoints, colors='k', linewidth=1, alpha=1))
            fig.colorbar(im).remove()

        gbezier = ghost_mesh.sample('bezier', 2)
        gpoints = gbezier.eval(self.geom).reshape(-1,2,domain.ndims)
        with export.mplfigure('ghost_mesh.png') as fig:
            ax = fig.add_subplot(111, aspect='equal', title='ghost mesh')
            ax.autoscale(enable=True, axis='both', tight=True)
            im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, numpy.zeros(points.shape[0]), shading='gouraud', cmap='binary')
            ax.add_collection(collections.LineCollection(gpoints, colors='k', linewidth=1, alpha=1))
            fig.colorbar(im).remove()

    def solution(self, domain, ns, sol):

        # Defining Paul Tol's `rainbow_PuBr` color map [https://personal.sron.nl/~pault/]
        clrs = ['#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
                '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
                '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
                '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
                '#DF4828', '#DA2222', '#B8221E', '#95211B', '#721E17',
                '#521A13']
        cmap = colors.LinearSegmentedColormap.from_list('rainbow_PuBr', clrs)
        cmap.set_bad('#FFFFFF')

        bezier = domain.sample('bezier', 2**self.maxrefine+1)
        points, uvals, pvals = bezier.eval([self.geom, ns.u, ns.p], **sol)

        with export.mplfigure('velocity.png') as fig:
            ax = fig.add_subplot(111, aspect='equal', title='velocity')
            ax.autoscale(enable=True, axis='both', tight=True)
            im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, numpy.linalg.norm(uvals, axis=-1), shading='gouraud', cmap=cmap)
            fig.colorbar(im)

        with export.mplfigure('pressure.png') as fig:
            ax = fig.add_subplot(111, aspect='equal', title='pressure')
            ax.autoscale(enable=True, axis='both', tight=True)
            im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, pvals, shading='gouraud', cmap=cmap)
            fig.colorbar(im)

# VTKWriter for plotting for 2D/3D cases
class VTKWriter(PostProcessor):

    def solution(self, domain, ns, sol):

        bezier = domain.sample('bezier', 2**self.maxrefine+1)
        points, uvals, pvals = bezier.eval([self.geom, ns.u, ns.p], **sol)
        export.vtk('solution', bezier.tri, points, u=uvals, p=pvals)
