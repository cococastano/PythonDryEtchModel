import numpy as np
import pyvista as pv
import matplotlib as mpl
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

# helper function s for cellular automata silicon ethcing simulation

def define_steps(recipe_steps, t_start, t_step):
    from collections import OrderedDict
    etch_grid = OrderedDict()
    etch_grid['init'] = []
    print('constructing specific step keys for topo container')
    for step in recipe_steps:
        for i_cycle, cycles in enumerate(range(recipe_steps[step]['cycles'])):
            # if it is a combined bosch-iso step
            if len(str(i_cycle)) < 2:
                if i_cycle == 9: 
                    i_cycle_str = str(i_cycle+1)
                else:
                    i_cycle_str = '0' + str(i_cycle+1)
                
            if recipe_steps[step]['bosch'] != None and \
            recipe_steps[step]['iso'] != None:
                # construct detailed keys for data container
                # i.e. key step1_bosch-iso6_bosch12_isotime100 is data for 
                # the 100th second if an iso etch following the 12th bosch step 
                # in the 6th cycle of a bosch-iso combined 1st step of the recipe
                # combined bosch-iso etching starts with a bosch step; the key
                # for this first step can be identified by an "_isotime0" flag
                for i_bosch in range(recipe_steps[step]['bosch']):
                    if len(str(i_bosch)) < 3:
                        if len(str(i_bosch)) == 1: 
                            if i_bosch == 9:
                                i_bosch_str = '0' + str(i_bosch+1)
                            else:
                                i_bosch_str = '00' + str(i_bosch+1)
                        elif len(str(i_bosch)) == 2: 
                            if i_bosch == 99:
                                i_bosch_str = str(i_bosch+1)
                            else:
                                i_bosch_str = '0' + str(i_bosch+1)
                                
                    # initial bosch cycle key
                    key = step + '_bosch-iso' + i_cycle_str + \
                          '_bosch' + i_bosch_str + '_isotime0'
                    etch_grid[key] = []
                for i_t,t in enumerate(range(t_start, 
                                             recipe_steps[step]['iso'], 
                                             t_step)):    
                    key = step + '_bosch-iso' + i_cycle_str + \
                          '_bosch' + i_bosch_str + '_isotime' + str(t+t_step)
                    etch_grid[key] = []
            elif recipe_steps[step]['bosch'] != None and \
            recipe_steps[step]['iso'] == None: 
                # similar key construction but specifically for bosch etching; it 
                # assumed that each cycle of bosch etching has the same etching
                # rate
                for i_bosch in range(recipe_steps[step]['bosch']):  
                    if len(str(i_bosch)) < 3:
                        if len(str(i_bosch)) == 1: 
                            if i_bosch == 9:
                                i_bosch_str = '0' + str(i_bosch+1)
                            else:
                                i_bosch_str = '00' + str(i_bosch+1)
                        elif len(str(i_bosch)) == 2: 
                            if i_bosch == 99:
                                i_bosch_str = str(i_bosch+1)
                            else:
                                i_bosch_str = '0' + str(i_bosch+1)
                    else: 
                        i_bosch_str = str(i_bosch+1)
                        
                    key = step + '_bosch' + i_bosch_str
                    etch_grid[key] = []
    
            elif recipe_steps[step]['bosch'] == None and \
            recipe_steps[step]['iso'] != None: 
                # similar key construction but specifically for iso etching; it is
                # possible to have multiple cycles of iso etching, i.e. each with
                # different conditions/rates
                for i_t,t in enumerate(range(t_start, 
                                             recipe_steps[step]['iso'], 
                                             t_step)):    
                    key = step + '_iso' + i_cycle_str + '_isotime' + \
                          str(t+t_step)
                    etch_grid[key] = []

    return etch_grid

def ion_source_dist(theta, sigma=1):
    J = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(theta-np.pi/2)**2/(2*sigma**2))
    return J

def etch_rate():
    k_b = 1.38e-23  # Boltzman constant [J/K]
    T_s = 100 + 274.15  # substrate temperature [K]
    k_0 = np.linspace(0,30,30)
    F_r = 150  # flow rate of SF6 [sccm]
    return
  
#def cross_section_slice(cells,p1=(-200,200),p2=(200,-200)):
#    if type(cells) == dict:
#        cells = np.array(list(cells.keys()))

def cross_section_slice(cells,cell_size,p1=(-200,200),p2=(200,-200)):
    from scipy import spatial
    if type(cells) == dict:
        cells = np.array(list(cells.keys()))
    elif type(cells) == set:
        cells 
    x_slice = np.arange(p1[0],p2[0],cell_size)
    m = (p2[1] - p1[1])/(p1[0] - p2[0])
    y_slice = m*(x_slice-p1[0]) + p1[1]
    xy = cells[:,:2]
    z = []
    x_out = []
    for x,y in zip(x_slice,y_slice):
        pt = np.asarray((x,y))
        dist = np.sum((xy-pt)**2,axis=1)
        ind = np.argmin(dist)
        z.append(cells[ind,2])
        x_out.append(x)

#    for x,y in zip(x_slice,y_slice):
#        dist,ind = spatial.KDTree(xy).query((x,y))
#        z.append(cells[ind,2])
#        x_out.append(x)
        
    return(x_out,z)
    
def compute_neigh_centers(cell,cell_size,wafer_thickness=500):
    # compute face neighbor cells
    x, y, z = cell[0], cell[1], cell[2]
    d = cell_size
    neighs = [[x-d, y, z],[x+d, y, z],
              [x, y-d, z],[x, y+d, z],
              [x, y, z-d],[x, y, z+d]]
    if z+d > wafer_thickness:
        neighs = neighs[:-1]
    return np.around(np.array(neighs),3)
    

def get_neighbor_cell_ids(grid, cell_idx):
    """helper to get neighbor cell IDs."""
    cell = grid.GetCell(cell_idx)
    pids = pv.vtk_id_list_to_array(cell.GetPointIds())
    neighbors = set(grid.extract_points(pids)['vtkOriginalCellIds'])
    neighbors.discard(cell_idx)
    return np.array(list(neighbors))

def plot_point_cloud(clouds,scalar='z'):
    exp_c = clouds[0]
    n_c = clouds[1]
    x_exp,y_exp,z_exp = exp_c[:,0],exp_c[:,1],exp_c[:,2]
    x_n,y_n,z_n = n_c[:,0],n_c[:,1],n_c[:,2]
    
    if scalar == 'z':
        cs_exp = z_exp
        cs_n = z_n
    else:
        cs_exp = scalar
        cs_n = z_n
    
    fig_exp = mpl.pyplot.figure()
    ax_exp = fig_exp.add_subplot(111, projection='3d')
    colors_map = 'rainbow'
    cm = mpl.pyplot.get_cmap(colors_map)
    c_norm = mpl.colors.Normalize(vmin=min(cs_exp), vmax=max(cs_exp))
    scalarMap = mpl.cm.ScalarMappable(norm=c_norm, cmap=cm)
    scalarMap.set_array(cs_exp)
    fig_exp.colorbar(scalarMap, shrink=0.5, aspect=5)
    ax_exp.scatter(x_exp, y_exp, z_exp, c=scalarMap.to_rgba(cs_exp))
    ax_exp.set_title('exposed points')

    fig_n = mpl.pyplot.figure()
    ax_n = fig_n.add_subplot(111, projection='3d')
    colors_map = 'rainbow'
    cm = mpl.pyplot.get_cmap(colors_map)
    c_norm = mpl.colors.Normalize(vmin=min(cs_n), vmax=max(cs_n))
    scalarMap = mpl.cm.ScalarMappable(norm=c_norm, cmap=cm)
    scalarMap.set_array(cs_n)
    fig_n.colorbar(scalarMap, shrink=0.5, aspect=5)
    ax_n.scatter(x_n, y_n, z_n, c=scalarMap.to_rgba(cs_n))
    ax_n.set_title('neighbor points')
    return

def plot_keys(container):
    x, y, z = [], [], []
    for key in container:
        if type(key) == str: key = eval(key)
        x.append(key[0])
        y.append(key[1])
        z.append(key[2])
        
    fig = mpl.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    
def remove_cells_by_state(grid, extract_idx):
    out_grid = grid.threshold([0,1], scalars='state', invert=False)
    out_grid.cell_arrays['on_surface'].dtype = bool
    out_grid.cell_arrays['exposed'].dtype = bool
    out_grid.cell_arrays['in_mask'].dtype = bool
    out_grid.cell_arrays['is_neighbor'].dtype = bool
    return out_grid

def make_cloud(container,is_cont_of_str=False):
    # helper function to make an np point cloud from cell center points
    # passed from dictionary keys or set of tuples
    # columns x, y, z, state, id where id = 1 is exposed and id = 0 is neighbor
    pt_clouds = []
    state = []
    iden = []
    for i_cont, cont in enumerate(container):
        if type(cont) == set and is_cont_of_str == False: 
            cont = list(cont)
        for i_cell, cell_center in enumerate(cont):
            if type(cell_center) == str: 
                cell_center = eval(cell_center)
            pt_clouds.append([cell_center[0], cell_center[1], cell_center[2]])
            try:
                state.append(cont[cell_center]['state'])
                iden.append(1)
            except:
                state.append(0)
                iden.append(0)
    out_cloud = np.array(pt_clouds)
    out_state = np.array(state)
    out_iden = np.array(iden)
    return(out_cloud,out_state,out_iden)

        

def make_grid(containers,cell_size):
    # helper function to make an unstructured grid from cell center points
    # passed from dictionary keys or set of tuples
    grids = []
    for container in containers:
        cells = []
        offset = np.arange(0,9*len(container),9)
        cell_type = np.array([pv.vtk.VTK_HEXAHEDRON]*len(container))
        point_count = 0
        pts = np.array([])
        if type(container) == set: container = list(container)
        for i_cell, cell_center in enumerate(container):
            cells.append(8)
            for p in range(8):
                cells.append(point_count)
                point_count += 1
            x = cell_center[0]
            y = cell_center[1]
            z = cell_center[2]
            d = cell_size/2
            # note this order
            cell_pts = np.array([[x-d,y-d,z-d],[x+d,y-d,z-d],
                                 [x+d,y+d,z-d],[x-d,y+d,z-d],
                                 [x-d,y-d,z+d],[x+d,y-d,z+d],
                                 [x+d,y+d,z+d],[x-d,y+d,z+d]])
            cell_pts = np.around(cell_pts,3)
            if pts.size == 0:
                pts = cell_pts
            else:
                pts = np.vstack((pts,cell_pts))
        # make unstructured grid
        grid = pv.UnstructuredGrid(offset,
                                   np.array(cells),
                                   cell_type,
                                   pts) 
        # make cell_arrays for state and neighbor label
        grid.cell_arrays['state'] = np.zeros(len(container))
        grid.cell_arrays['neighbor'] = [False] * len(container)
        # assuming exposed cells with a state are in a dict
        if type(container) == dict:
            for i_cell,cell_center in enumerate(container):
                grid.cell_arrays['state'][i_cell] = \
                container[cell_center]['state']
        # and type set (now list) is the neighbor cells
        elif type(container) == list:
            for i_cell,cell_center in enumerate(container):
                grid.cell_arrays['neighbor'][i_cell] = True
        # add to grids list
        grids.append(grid) 
    # now merge grids together
    if len(grids) == 2:
        out_grid = grids[0].merge(grids[1],merge_points=False,
                        main_has_priority=True)
    elif len(grids) == 1:
        out_grid = grids[0]
    else:
        out_grid = grids[0]
        for grid in grids[1:]:
            out_grid = out_grid.merge(grid,merge_points=False,
                        main_has_priority=True)
            
    return out_grid
                
        

def plot_subset_index_cells(grid,i_cell,i_neighbors=np.array([])):
    plotter = pv.BackgroundPlotter()
    plotter.add_mesh(grid.extract_all_edges(), color='k', label='whole mesh')
    if i_neighbors.size != 0:
        plotter.add_mesh(grid.extract_cells(i_neighbors), color=True, 
                         opacity=0.5, label='neighbors')
    plotter.add_mesh(grid.extract_cells(i_cell), color='pink', 
                     opacity=0.75, label='the cell')
    plotter.add_legend()
    plotter.show()
    return

def plot_subset_cells(grid,subset=None,scalar='z',invert=False):
    if invert == True: 
        flag_bool = False
        flag_int = 0
    else:
        flag_bool = True
        flag_int = 1
    if subset != None:
        extract_idx = []
        for i_cell in range(grid.n_cells):
            if grid.cell_arrays[subset][i_cell] == flag_bool \
            or grid.cell_arrays[subset][i_cell] == flag_int:
                extract_idx.append(i_cell)
        plot_grid = grid.extract_cells(extract_idx)
    else:
        plot_grid = grid
    plotter = pv.BackgroundPlotter()
    if scalar == 'z':
        s = plot_grid.points[:,2]
    else:
        s = plot_grid.cell_arrays[scalar]
    
    plotter.add_mesh(plot_grid, show_edges=True, scalars=s)
    plotter.add_scalar_bar()
    return    


def is_in_mask(x,y,mask_paths,radius=0.0,alt_label=False):
    # True for x, y pt coord in mask contour
    # False for x, y pt coord not in mask contour
    # SurfBound for pt on wafer surface as the boundary of an etch

    try:
        # one point, multiple mask paths
        for path in mask_paths:
            if alt_label != False:
                inside = alt_label
                break            
            elif alt_label == False: 
                inside = mask_paths[path].contains_point((x,y),radius=radius)
                if inside == True: break
    except:
        # multiple points, one mask path
        if type(x) is np.ndarray:
            if alt_label == False:
                inside = [mask_paths.contains_point((i,j),radius=radius) \
                          for i,j in zip(x,y)]
            elif alt_label != True:
                inside = [alt_label for i,j in zip(x,y)]
        else:
            if alt_label == False:
                inside = mask_paths.contains_point((x,y),radius=radius)
            elif alt_label != False:
                inside = alt_label
    return inside


def fix_norm(pt, norm, ref_pt=np.array([0,0,0])):
    # determine angle between unit vector and vector between point and 
    # reference point
    ref_vec = np.array([pt[0] - ref_pt[0], 
                        pt[1] - ref_pt[1], 
                        pt[2] - ref_pt[2]])
    ref_vec = ref_vec / np.linalg.norm(ref_vec)
    angle1 = np.arccos(np.dot(ref_vec,norm))
    angle2 = np.arccos(np.dot(ref_vec,-1*norm))

    if angle2 > angle1: 
        pass
    else:
        norm = -1*norm
    return norm

def compute_normals(points, use_nn=False, flat=False, 
                    ref_pt=np.array([0,0,0])):

    if flat == True:  # add dummy layer to make 3D
        knns = NearestNeighbors(n_neighbors=use_nn).fit(points)
        dists, indices = knns.kneighbors(points, return_distance=True)
        dummy_dist = np.mean(dists)
        dummy_upper_layer = np.copy(points)
        dummy_lower_layer = np.copy(points)
        dummy_lower_layer[:,-1] -= dummy_dist
        dummy_upper_layer[:,-1] += dummy_dist
        dummy_layers = np.vstack((dummy_upper_layer, dummy_lower_layer))
        points = np.vstack((points,dummy_layers))
    
    
#    fig = mpl.pyplot.figure()
#    ax = fig.add_subplot(111, projection='3d')
    if type(points) == dict:
        pts = [list(cell) for cell in points]
        cloud = make_cloud([exposed_cells])[0]
        cloud = np.vstack((cloud,pts))
        knns = NearestNeighbors(n_neighbors=use_nn).fit(cloud)
        dists, indices = knns.kneighbors(cloud, return_distance=True)
        unit_norms = {}
        for pt in list(points.keys()):
            x, y, z = pt[0], pt[1], pt[2]
            nns = []#[[x, y, z]]
            # collect coordinates of nns
            pt_i = np.where((cloud[:,0] == x) & (cloud[:,1] == y) 
                            & (cloud[:,2] == z))[0]
            pt_i = pt_i[0]
            for nn in range(use_nn):
                nns.append(cloud[indices[pt_i][nn],:])
            nns = np.array(nns)
            
            # compute centroid and shift points relative to it
            cent = np.mean(nns, axis=0)
            xyzR = nns - cent
            u, sigma, v = np.linalg.svd(xyzR)
            unit_norm = v[2] / np.linalg.norm(v[2])
            unit_norms[pt] = unit_norm
            
    elif points.shape[0] == 3:
        cloud = make_cloud([exposed_cells])[0]
        knns = NearestNeighbors(n_neighbors=use_nn).fit(cloud)
        dists, indices = knns.kneighbors(cloud, return_distance=True)
        pt_i = np.where((cloud[:,0] == points[0]) & (cloud[:,1] == points[1]) 
                        & (cloud[:,2] == points[2]))[0]
        if pt_i.shape[0] == 0:
            cloud = np.vstack((cloud,points))
            knns = NearestNeighbors(n_neighbors=use_nn).fit(cloud)
            dists, indices = knns.kneighbors(cloud, return_distance=True)
            pt_i = np.where((cloud[:,0] == points[0]) & (cloud[:,1] == points[1]) 
                            & (cloud[:,2] == points[2]))[0]
        nns = []
        for nn in range(use_nn):
            nns.append(cloud[indices[pt_i][0][nn],:])
        nns = np.array(nns)
        cent = np.mean(nns, axis=0)
        xyzR = nns - cent
#        xyzRT = np.transpose(xyzR)
        # compute singular value decomposition 
        u, sigma, v = np.linalg.svd(xyzR)
        unit_norms = v[2] / np.linalg.norm(v[2])
        
    else:
        knns = NearestNeighbors(n_neighbors=use_nn).fit(points)
        dists, indices = knns.kneighbors(points, return_distance=True)
        unit_norms = []
        for pt_i, pt in enumerate(points):
            x, y, z = pt[0], pt[1], pt[2]
            nns = []#[[x, y, z]]
            # collect coordinates of nns
            for nn in range(use_nn):
                nns.append(points[indices[pt_i][nn],:])
            nns = np.array(nns)
            
            # compute centroid and shift points relative to it
            cent = np.mean(nns, axis=0)
            xyzR = nns - cent
    #        xyzRT = np.transpose(xyzR)
            # compute singular value decomposition 
            u, sigma, v = np.linalg.svd(xyzR)
            unit_norm = v[2] / np.linalg.norm(v[2])
            unit_norms.append(unit_norm)
            # determine angle between unit vector and vector between point and 
            # reference point
    #        ref_vec = np.array([x - ref_pt[0], y - ref_pt[1], z - ref_pt[2]])
    #        ref_vec = ref_vec / np.linalg.norm(ref_vec)
    
    #        angle1 = np.arccos(np.dot(ref_vec,unit_norm))
    #        angle2 = np.arccos(np.dot(ref_vec,-1*unit_norm))
            
    #        if angle2 > angle1: 
    #            unit_norms.append(unit_norm)
    #        else:
    #            unit_norms.append(-1*unit_norm)
        unit_norms = np.array(unit_norms)
    return unit_norms

def compute_angle(norm,ref_pt=[0,0,0],wafer_thickness=500):
    ref_vec = np.array([ref_pt[0] - ref_pt[0],
                        ref_pt[1] - ref_pt[1], 
                        wafer_thickness + 10 - ref_pt[2]])
    ref_vec = ref_vec / np.linalg.norm(ref_vec)
    angle1 = np.arccos(np.dot(ref_vec,norm))
    angle2 = np.arccos(np.dot(ref_vec,-1*norm))
    if angle2 > angle1: 
        angle = angle1
    else:
        angle = angle2
    return angle

        
def normal_from_neighbors(cell_tuple,removed_cells,cell_size,n_cells_span=2,
                          wafer_thickness=500):
    x, y, z = cell_tuple[0], cell_tuple[1], cell_tuple[2]
    x_c = np.linspace(x-n_cells_span*cell_size,
                      x+n_cells_span*cell_size,2*n_cells_span+1)
    y_c = np.linspace(y-n_cells_span*cell_size,
                      y+n_cells_span*cell_size,2*n_cells_span+1)
    z_c = np.linspace(z-n_cells_span*cell_size,
                      z+n_cells_span*cell_size,2*n_cells_span+1)
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_c, y_c, z_c)
    
    radius = (cell_size/2) * np.sqrt(24) + (cell_size/2)*0.1
#    radius = ((cell_size) * np.sqrt(5)) + (cell_size/2)*0.2

    normal_vect = [0,0,0]

    for i in range(len(x_c)):
       for j in range(len(y_c)):
           for k in range(len(z_c)):
               x_v = round(x_mesh[i,j,k],3)
               y_v = round(y_mesh[i,j,k],3)
               z_v = round(z_mesh[i,j,k],3)
               if (str((x_v,y_v,z_v)) in removed_cells and \
                   z_v <= wafer_thickness):
                    d = np.sqrt((x-x_v)**2 + (y-y_v)**2 + (z-z_v)**2)
                    if d < radius:
                        normal_vect[0] += round(x_v - x,3)
                        normal_vect[1] += round(y_v - y,3)
                        normal_vect[2] += round(z_v - z,3)
    if np.linalg.norm(normal_vect) == 0: 
        normal_vect = compute_normals(np.array(list(cell_tuple)), use_nn=8, 
                    ref_pt=np.array([0,0,0]))  
    unit_norm = normal_vect / np.linalg.norm(normal_vect)
    return(unit_norm)
    
def compute_normals_for_cells(exposed_cells,removed_cells,
                              cell_size,n_cells_span=2,
                              wafer_thickness=500):
    
    radius = (cell_size/2) * np.sqrt(24) + (cell_size/2)*0.102
    normal_vects = {}
    alt_method_points = {}
    alt_method_flag = False
    n_cells = len(exposed_cells)
    n_normals = 0
    try:
        keys = list(exposed_cells.keys())
    except:
        keys = [exposed_cells]
        
    for i_cell, cell in enumerate(keys):
        if i_cell%(int(n_cells/5)) == 0: 
            print('\t\tcomputing normal for pt %i of %i' %(i_cell,n_cells))
        try:
            need_norm = exposed_cells[cell]['need_norm']
        except:
            need_norm = True
        if need_norm == True:
            n_normals += 1
            x, y, z = cell[0], cell[1], cell[2]
            x_c = np.linspace(x-n_cells_span*cell_size,
                              x+n_cells_span*cell_size,2*n_cells_span+1)
            y_c = np.linspace(y-n_cells_span*cell_size,
                              y+n_cells_span*cell_size,2*n_cells_span+1)
            z_c = np.linspace(z-n_cells_span*cell_size,
                              z+n_cells_span*cell_size,2*n_cells_span+1)
            x_mesh, y_mesh, z_mesh = np.meshgrid(x_c, y_c, z_c)
            normal_vect = [0,0,0]
            for i in range(len(x_c)):
               for j in range(len(y_c)):
                   for k in range(len(z_c)):
                       x_v = round(x_mesh[i,j,k],3)
                       y_v = round(y_mesh[i,j,k],3)
                       z_v = round(z_mesh[i,j,k],3)
                       if (str((x_v,y_v,z_v)) in removed_cells and \
                           z_v <= wafer_thickness):
                            d = np.sqrt((x-x_v)**2 + (y-y_v)**2 + (z-z_v)**2)
                            if d < radius:
                                normal_vect[0] += round(x_v - x,3)
                                normal_vect[1] += round(y_v - y,3)
                                normal_vect[2] += round(z_v - z,3)
            if np.linalg.norm(normal_vect) == 0: 
                alt_method_flag = True
                alt_method_points[cell] = []
            else:
                normal_vects[cell] = (normal_vect / np.linalg.norm(normal_vect))
    if alt_method_flag == True:
        alt_method_points = compute_normals(alt_method_points, use_nn=8)
    
    print('\t\t\tassigning normals to %i cells' % n_normals)
    for cell in list(exposed_cells.keys()):
        if exposed_cells[cell]['need_norm'] == True:
            try:
                exposed_cells[cell]['normal'] = normal_vects[cell]
            except:
                exposed_cells[cell]['normal'] = alt_method_points[cell]
            exposed_cells[cell]['need_norm'] = False
    return exposed_cells