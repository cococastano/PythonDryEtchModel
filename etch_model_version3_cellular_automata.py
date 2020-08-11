import numpy as np
import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pyvista as pv
import open3d as o3d
import shapely.geometry as geometry
import time
#from skimage import measure
from ast import literal_eval
from copy import deepcopy
from scipy.spatial import distance as dist
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from matplotlib.path import Path
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union, polygonize
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from etch_sim_utilities import *

nps = pv.vtk.util.numpy_support

mpl.style.use('default')

start_time = time.time()

"""
Written by Nicolas Castano

Model continuous etching into silicon wafer on the PT-DSE tool in the SNF 
based known etch rates.

Data is stored in ordered dictionary with keys as specific step, kept in the
order in which it was created:
    etch_grid = {'init': [pv.PolyData(mask_cont_0), pv.PolyData(mask_cont_1),
                          pv.PolyData(mask_cont_2), ...],
                 global_step_0: [pv.PolyData(mask_cont_0), ne
                                 pv.PolyData(mask_cont_1),
                                 pv.PolyData(mask_cont_2), ...],                 
                 global_step_N: [pv.PolyData(mask_cont_0)]}
    
"""

    
###############################################################################
################################ USER INPUTS ##################################
###############################################################################

# define recipe
# ex: {'step1':{'bosch':13,'iso':100,'cylces':7},
#      'step2':{'bosch':240,'iso':None,'cycles':240},
#      'step3':{'bosch':None,'iso':70,'cycles':1}}
#recipe_steps = {'step1':{'bosch':13,'iso':100,'cycles':7},
#                'step2':{'bosch':240,'iso':None,'cycles':240},
#                'step3':{'bosch':None,'iso':70,'cycles':1}}
#recipe_steps = {'step0':{'bosch':7,'iso':5,'cycles':2}}
#recipe_steps = {'step01':{'bosch':15,'iso':100,'cycles':7},
##                'step02':{'bosch':300,'iso':None,'cycles':300},
#                'step03':{'bosch':None,'iso':100,'cycles':1}}
recipe_steps = {'step01':{'bosch':20,'iso':125,'cycles':7},
#                'step02':{'bosch':300,'iso':None,'cycles':300},
                'step03':{'bosch':None,'iso':100,'cycles':1}}


# load mask
im_dir = 'C:/Users/nicas/Documents/E241-MicroNanoFab/masks/'
im_file = 'mask5_R5_C3_v0.png'
pixel_um_conv = 49.291/80.4384
# 151.37/100  # for R2_C2
# for R5_C3: 49.291/80.4384  # px/um

# read in mask image and define contour
im_path = im_dir + im_file
curr_im = cv2.imread(im_path, cv2.IMREAD_ANYDEPTH)   
curr_im = cv2.GaussianBlur(curr_im,(3,3),0)
rgb_im = cv2.cvtColor(curr_im, cv2.COLOR_GRAY2RGB)
cont_im, conts, hier = cv2.findContours(curr_im, cv2.RETR_LIST, 
                                        cv2.CHAIN_APPROX_NONE)    
conts_im = cv2.drawContours(rgb_im, conts, -1, (0,255,0),3)
# show the contour to verify 
dummy_i = im_file.find('.png')
out_file = im_dir + im_file[:dummy_i] + '_out' + im_file[dummy_i:]
cv2.imwrite(out_file, conts_im)

cell_size = 5 # microns
wafer_thickness = 500  # microns

t_start = 0 # seconds
t_step = 5

h = curr_im.shape[0]
w = curr_im.shape[1]

contour_read_step = 5
topo_im = np.zeros_like(curr_im)
norm_span = 7  # span of data points taken for computing normals
window_len = 17  # for smoothing of mask contour read
horiz_to_vert_rate_ratio = 0.6
def vert_rate(z):
    a = 0.141
    b = 0.0007
    return a*np.exp(-b*z)
def horiz_rate(z):
    return 0.8#0.8*vert_rate(z)

#vert_rate = 8.5/60  # um/s
def bosch_vert_step(z):
    return 0.84 - 0.1/500*z
    
#bosch_vert_step = 0.84  # um/step

#horiz_rate = 0.09# vert_rate*0.6#90/600 # vert_rate*0.6  # um/s

# advanced settings
set_res = 3000  # resolution of vtk plane (mesh density)
cmap = 'gnuplot'  # 'inferno' 'viridis'  # 'hot'
vmin = -290  # expected range of depth for color bar (min)
vmax = 0
# for plotting srface plot
rstride = 2
cstride = 2


###############################################################################
###############################################################################
###############################################################################


# initialize global topo data container following data structure 
# indicated in the script header
# construct global data container; this is a ordered dictionary so later we
# can loop over keys and ensure that 
etch_grid = define_steps(recipe_steps, t_start, t_step)                        
n_steps = len(list(etch_grid.keys()))


# construct mask paths and check is cell centers are within masks    
# path objects used for determining if point is within mask
mask_paths = {}
# build initial geometries from mask that will be tracked through solution
print('building initial features')
x_min, x_max, y_min, y_max = 10**8, -10**8, 10**8, -10**8 
for c, cont in enumerate(conts):
    x = []
    y = []
    # gather points in mask contours
    for p, point in enumerate(cont):
        if p%contour_read_step == 0:
            # translate point so mask centered at 0,0
            temp_x = point[0][0]/pixel_um_conv - (w/pixel_um_conv)/2
            temp_y = point[0][1]/pixel_um_conv - (h/pixel_um_conv)/2
            x.append(temp_x)
            y.append(temp_y)   
            if temp_x < x_min: x_min = round(temp_x,3)
            if temp_y < y_min: y_min = round(temp_y,3)
            if temp_x > x_max: x_max = round(temp_x,3)
            if temp_y > y_max: y_max = round(temp_y,3)

    # force last point to be on top of first point to close the polygon
    # remove redundant points
    x[-1] = x[0]
    y[-1] = y[0]
    # smooth contour with spline
    points = np.hstack((np.reshape(np.array(x),[len(x),1]),
                       np.reshape(np.array(y),[len(y),1])))        
    tck, u = splprep(points.T, u=None, s=0.0, per=1) 
    u_new = np.linspace(u.min(), u.max(), len(cont))
    x_spline, y_spline = splev(u_new, tck, der=0)
    points = np.hstack((np.reshape(np.array(x_spline),[len(x_spline),1]),
                        np.reshape(np.array(y_spline),[len(y_spline),1])))
    # make polygon objects
    mask_poly = Polygon(points)  
    # make path objects (has nice contains_points method)
    mask_paths[c] = Path(mask_poly.exterior,closed=True)
    # this just means theres no buffer region around the feature
    buff_mask = mask_paths[c]  
    
    
    
# initialize the starting grid that will be etched away using cellular 
# automata method
x_axis = np.arange(x_min-cell_size,x_max+cell_size,cell_size)
y_axis = np.arange(y_min-cell_size,y_max+cell_size,cell_size)
z_axis = np.array([wafer_thickness-cell_size,
                   wafer_thickness])

x_nodes, y_nodes, z_nodes = np.meshgrid(x_axis, y_axis, z_axis)
init_grid = pv.StructuredGrid(x_nodes, y_nodes, z_nodes)
init_grid = init_grid.cast_to_unstructured_grid()
    
start_time = time.time()
# initialize the cell dictionaries for 'exposed_cells'
#   key: tuple of cell center coord
#   value: {'state': 0 to 1, 'in_mask': True or False, 
#           'neighbors':[(coord neigh 0),(coord neigh 1),...]}
# and for 'neighbor_cells'
#   this is a set that contains tuples of coords of cells that are added to 
#   be neighbors of exposed cells
exposed_cells = {}
neighbor_cells = set()
removed_cells = set()
known_in_mask_coords = set()
# get cell centers for finding cell that are in the mask
cell_centers = np.around(np.array(init_grid.cell_centers().points),3)
n_cells= cell_centers.shape[0]
for i_cell, pt in enumerate(cell_centers):
    if i_cell % int(n_cells/10) == 0:
        print('    finding exposed and neighbor cells: %i of %i' % \
              (i_cell,n_cells-1))
    temp_cell = init_grid.GetCell(i_cell)
    in_mask = is_in_mask(pt[0],pt[1],mask_paths)
    if in_mask == True:
        temp_tuple = tuple(pt[0:2])
        # had to use str here for this to work
        known_in_mask_coords.add(str(temp_tuple))  
        
        cell_tuple = tuple(pt)
        exposed_cells[cell_tuple] = {'state':1, 'in_mask':in_mask, 
                                    'neighbors':[],'normal':[],
                                    'need_norm':True}
        neighs = compute_neigh_centers(pt,cell_size)
        for neigh in neighs:
            # store neighbor if its not a surface cell or its outside of paths
            if ((neigh[2] != (wafer_thickness - cell_size/2)) or \
                (neigh[2] == (wafer_thickness - cell_size/2) and \
                 is_in_mask(neigh[0], neigh[1], mask_paths) == False)):
                neigh_tuple = tuple(neigh)
                exposed_cells[cell_tuple]['neighbors'].append(neigh)            
                # only store neighbors on lower layer or if neighbor outside
                # of mask
                neighbor_cells.add(neigh_tuple)
            else:
                removed_cells.add(str(cell_tuple))
                
# compute neighbors from neeighbors
for cell_tuple in exposed_cells:
    exposed_cells[cell_tuple]['normal'] = np.array([0,0,1])#normal_from_neighbors(cell_tuple,
#                                                      removed_cells,
#                                                      cell_size=cell_size)
                
                
## extract all exposed_cells and neighbor_cells to the same unstructured grid
#init_grid = make_grid([exposed_cells,neighbor_cells], cell_size)
    
                
# define the grid for the initial step      
print('making init grid')  
etch_grid['init'] = make_cloud((exposed_cells, neighbor_cells))

    
print('-------- %.2f seconds ---------' % (time.time()-start_time))


step_index_lookup = {i:key for i,key in enumerate(etch_grid)}
print('')
# loop over etch_grid keys (after init) which represent each detailed step
loop_steps = [key for key in list(etch_grid.keys()) if 'init' not in key]
curr_process = 'init'
extract_flag = False
d = cell_size
center_to_surface = wafer_thickness - cell_size/2
diag_dist = round(np.sqrt(2*cell_size**2),3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax = plt.subplot(111,projection='polar')
plot_flag = True

for step_i, step in enumerate(loop_steps,start=1):
    
    #plot some results
    if step_i % (int(len(loop_steps)/10)) == 0:
        print('making plot or %s' % step)
        neigh_pts,neigh_states,_ = make_cloud([neighbor_cells])
        neigh_obj = pv.PolyData(neigh_pts)
        plotter = pv.BackgroundPlotter()
        plotter.isometric_view()
        plotter.add_mesh(neigh_obj, show_edges=True,
                         scalars=neigh_obj.points[:,2],
                         point_size=8,
                         render_points_as_spheres=True,
                         cmap='inferno',
                         clim=[min(neigh_obj.points[:,2]), 
                               wafer_thickness])
        plotter.add_scalar_bar(title='z height',height=0.08,width=0.4,
                               position_x=0.01,position_y=0.01)
        plotter.add_text(step)
        
        
#    if dummy_i != len(etch_grid)-1: step_i = dummy_i + 1
    master_step = step.split('_')[0]
    
    # get list of features from last step to etch in current step
    curr_grid = etch_grid[step_index_lookup[step_i-1]]
    
    # define current process for printing and switching
    prev_process = curr_process
    if 'bosch-iso' in step:
        if 'isotime0' in step:
            curr_process = 'bosch-iso: bosch'
        else:
            curr_process = 'bosch-iso: iso'
    else:
        if 'bosch' in step.split('_')[1]:
            curr_process = 'bosch'
        elif 'iso' in  step.split('_')[1]:
            curr_process = 'iso'
    if curr_process != prev_process:
        print('current process: %s \t step %i of %i'
              %(step, step_i,n_steps))

    # if extract_flag was activated
    if extract_flag == True:
        extract_flag = False
        print('\t\tcollecting neighbors and deleting cells')
        del_cells = []  # for deleting later
        convert_neighbors = set()
        new_cell_neighbors = {}
        
        # flag cells to be deleted from exposed_cells and store removed cells
        for cell_center in exposed_cells:
            if exposed_cells[cell_center]['state'] < 0:
                del_cells.append(tuple(cell_center))  # collect cells to delete
                # store removed cells
                removed_cells.add(str(tuple(cell_center)))
            elif np.isnan(exposed_cells[cell_center]['state']):
                del_cells.append(tuple(cell_center))
                print('\t\tremoving nan cells')
                        
        # collect neighbors to be converted to exposed cells
        for cell_center in exposed_cells:
            if exposed_cells[cell_center]['state'] < 0:                
                for neigh_center in exposed_cells[cell_center]['neighbors']:
                    if str(tuple(neigh_center)) not in removed_cells:
                        convert_neighbors.add(tuple(neigh_center))
                
                
        
        # remove from exposed_cells
        for cell in del_cells: del exposed_cells[cell]
        
        print('\t\tconverting neighbor cells to exposed')
        # add converted neighbors to exposed_cells container
        for i_cell, new_cell in enumerate(list(convert_neighbors)):
            # collect neighbors for new cell
            temp_tuple = tuple([list(new_cell)[0], list(new_cell)[1]])
            coord_tuple = str(temp_tuple)
            if coord_tuple in known_in_mask_coords:
                label = True
            else:
                label = False#is_in_mask(x,y,mask_paths)
            exposed_cells[new_cell] = {'in_mask':label, 'state':1,
                                       'neighbors':[],'normal':[],
                                       'need_norm':True}

        # with only face neighbors, all neighbors of exposed cells
        # should be now exposed
        for convert_neigh in list(convert_neighbors):
            new_cell_neighbors[tuple(convert_neigh)] = []
            # determine new neighbors
            x, y, z = convert_neigh[0], convert_neigh[1], convert_neigh[2]
            new_cell_neigh_centers = \
                compute_neigh_centers(np.array(convert_neigh),cell_size)
            for neigh_center in new_cell_neigh_centers:
                temp_tuple = tuple(neigh_center)
                if (str(temp_tuple) not in removed_cells and \
                    temp_tuple not in exposed_cells and \
                    neigh_center[2] != center_to_surface):
                    
                    neighbor_cells.add(temp_tuple)
                    
                new_cell_neighbors[tuple(convert_neigh)].append(neigh_center)
                        # store neigh for each added cell in dict
        # add converted neighbors to exposed_cells container
        for i_cell, new_cell in enumerate(convert_neighbors):
            # collect neighbors for new cell
            neighs = [n for n in new_cell_neighbors[new_cell]]
            exposed_cells[tuple(new_cell)]['neighbors'] = neighs
            try:
                neighbor_cells.remove(new_cell)
            except:
                pass
        # compute normals for exposed cells
        exposed_cells = compute_normals_for_cells(exposed_cells,removed_cells,
                                                  cell_size)


    if (curr_process == 'bosch-iso: bosch' or curr_process == 'bosch'):
        # bosch step is a vertical etch of exposed_cells in the x, y bounds
        # of the mask ('in_mask' == True)
        n_bosch_steps = recipe_steps[master_step]['bosch']
        if curr_process == 'bosch-iso: bosch':
            curr_bosch_step = int(step.split('_')[-2].split('bosch')[-1])
        else:
            curr_bosch_step = int(step.split('_')[-1].split('bosch')[-1])
            

        print('\tbosch step %i of %i' % \
              (curr_bosch_step, n_bosch_steps))
            
        for cell_center in exposed_cells:
            if exposed_cells[cell_center]['in_mask'] == True:
                exposed_cells[cell_center]['state'] -= bosch_vert_step(cell_center[2])/ \
                                                       cell_size
                if (extract_flag == False and \
                    exposed_cells[cell_center]['state'] < 0):
                    extract_flag = True
                
       
        # update next step topo
        grid_plot = make_cloud((exposed_cells, neighbor_cells))
        etch_grid[step] = grid_plot
        
    elif (curr_process == 'bosch-iso: iso' or curr_process == 'iso'):
        # bosch step is a vertical etch of all exposed_cells
        
        n_iso_steps = recipe_steps[master_step]['iso']
        curr_iso_step = int(step.split('_')[-1].split('isotime')[-1])

        print('\tiso time %i of %i seconds' % \
              (curr_iso_step, n_iso_steps))
            
        # compute normals
        exp_pts = make_cloud([exposed_cells])[0]
#        exp_norms = compute_normals(exp_pts,use_nn=5,
#                                    ref_pt=np.array([np.mean(exp_pts[:,0]),
                                                     
        
        angles = []
        amounts = []
        x_center, y_center, z_center = [],[],[]
        for i_cell, cell_center in enumerate(exposed_cells):
            angle = compute_angle(exposed_cells[cell_center]['normal'],
                                  ref_pt=np.array([np.mean(exp_pts[:,0]),
                                                   np.mean(exp_pts[:,1]),
                                                   np.mean(exp_pts[:,2])]))
            curr_vert_rate = vert_rate(cell_center[2])
            curr_horiz_rate = horiz_rate(cell_center[2])
            etch_amount = (curr_vert_rate*t_step)*np.cos(angle) + \
                          (curr_horiz_rate*t_step)*np.sin(angle)
#            etch_amount = (horiz_rate*t_step)*np.cos(angle)
            exposed_cells[cell_center]['state'] -= etch_amount/cell_size
            
            if (extract_flag == False and \
                exposed_cells[cell_center]['state'] < 0):
                extract_flag = True
            if step_i > 30 and plot_flag == True:
                angles.append(angle)
                amounts.append(etch_amount)
                x_center.append(cell_center[0])
                y_center.append(cell_center[1])
                z_center.append(cell_center[2])
                
        if step_i > 30 and plot_flag == True:
            print('plotting!')
            colors_map = 'inferno'
            cm = plt.get_cmap(colors_map)
            c_norm = mpl.colors.Normalize(vmin=min(amounts), vmax=max(amounts))
            scalarMap = cmx.ScalarMappable(norm=c_norm, cmap=cm)
            scalarMap.set_array(amounts)
            fig.colorbar(scalarMap, shrink=0.5, aspect=5)
            ax.scatter(x_center, y_center, z_center,c=scalarMap.to_rgba(amounts))
            plot_flag = False
            
        # update next step topo
        grid_plot = make_cloud((exposed_cells, neighbor_cells))
        etch_grid[step] = grid_plot
        
    else:
        etch_grid[step] = make_cloud((exposed_cells, neighbor_cells))
        extract_flag = False
        pass

        
print('-------- %.2f seconds ---------' % (time.time()-start_time))

        
exposed_pts,exposed_states,_ = make_cloud([exposed_cells])
neigh_pts,neigh_states,_ = make_cloud([neighbor_cells])
plot_point_cloud((exposed_pts,neigh_pts),scalar='z')

with_data = [g for g in etch_grid if len(etch_grid[g]) != 0]
plot_png_dir = 'C:/Users/nicas/Documents/E241-MicroNanoFab/codes/' + \
               'etch_model_version5_1/'
               
#dict_file = plot_png_dir + 'exposed_cells_mask5_R5_C3.txt'
## save exposed_cell to file
#with open(dict_file, 'w') as file:
#     file.write(json.dumps(exposed_cells))
# save vtk file
exposed_obj = pv.PolyData(exposed_pts)
neigh_obj = pv.PolyData(neigh_pts)
vtk_save_exp_obj = plot_png_dir + 'exposed_obj.vtk'
vtk_save_neigh_obj = plot_png_dir + 'neigh_obj.vtk'

exposed_obj.save(vtk_save_exp_obj)
neigh_obj.save(vtk_save_neigh_obj)

#pcd_exp = o3d.geometry.PointCloud()
#pcd_neigh = o3d.geometry.PointCloud()
#pcd_exp.points = o3d.utility.Vector3dVector(exposed_pts)
#pcd_neigh.points = o3d.utility.Vector3dVector(neigh_pts)
#o3d.visualization.draw_geometries([pcd_exp,pcd_neigh])

#exp_norms = []
#exp_pts = etch_grid['step01_bosch-iso02_bosch020_isotime5'][0]
##compute_normals_for_cells(exp_pts)
#
#for i,pt in enumerate(exp_pts):
#    if i%1000 == 0: print('%i of %i' %(i, len(exp_pts)))
#    try:
#        norm = exposed_cells[tuple(pt)]['normal']
#    except:
#        norm = normal_from_neighbors(tuple(pt),removed_cells,cell_size)
#    if len(norm) == 0: norm = normal_from_neighbors(tuple(pt),
#                                                    removed_cells,cell_size)
#    exp_norms.append(norm)
#pcd_exp = o3d.geometry.PointCloud()
#pcd_exp.points = o3d.utility.Vector3dVector(exp_pts)
#for norm in exp_norms:
#    pcd_exp.normals.append(norm)
#o3d.visualization.draw_geometries([pcd_exp])



select_data = with_data[0::int(len(with_data)/len(with_data))]
for i_step,step in enumerate(loop_steps):

    if i_step % 5 == 0: 
        print('writing plot .png file %i of %i' % (i_step+1,len(select_data)))
    
    pts = etch_grid[step][0]
    states = etch_grid[step][1]
    iden = etch_grid[step][2]
    
    exp_cell_idx = np.where(iden == 1)[0]
    neigh_cell_idx = np.where(iden == 0)[0]
    
    # unpack
    exp_cells,exp_states = pts[exp_cell_idx],states[exp_cell_idx]
    neigh_cells,neigh_states = pts[neigh_cell_idx],states[neigh_cell_idx]
    
    exposed_obj = pv.PolyData(exp_cells)
    neigh_obj = pv.PolyData(neigh_cells)

    plotter = pv.Plotter(off_screen=True)
    plotter.isometric_view()
    
#    plotter.add_mesh(neigh_obj, show_edges=True,
#                     scalars=neigh_obj.points[:,2],
#                     point_size=8,
#                     render_points_as_spheres=True,
#                     cmap='inferno',
#                     clim=[143, 500])
#    plotter.add_scalar_bar(title='z height',height=0.08,width=0.4,
#                           position_x=0.01,position_y=0.01)
    plotter.add_text(step)

    plotter.add_mesh(exposed_obj, show_edges=True,
                     scalars=exposed_obj.points[:,2],
                     point_size=8,
                     render_points_as_spheres=True,
                     cmap='rainbow',
                     clim=[min(exposed_obj.points[:,2]), wafer_thickness])
    plotter.add_scalar_bar(title='z_height',height=0.08,width=0.4,
                           position_x=0.01,position_y=0.1)
    
    file_name = plot_png_dir + step + '.png'
    
    plotter.screenshot(file_name,transparent_background=True)


with_data = [g for g in etch_grid if len(etch_grid[g]) != 0]
select_data = with_data[-2:-1:1]#int(len(with_data)/10)]
for ind,step in enumerate(select_data):#range(20):
    pts = etch_grid[step][0] 
    states = etch_grid[step][1]
    iden = etch_grid[step][2]  
    
    exp_cell_idx = np.where(iden == 1)[0]
    neigh_cell_idx = np.where(iden == 0)[0]
    
    exp_cells,exp_states = pts[exp_cell_idx],states[exp_cell_idx]
    neigh_cells,neigh_states = pts[neigh_cell_idx],states[neigh_cell_idx]
    
    cells = pv.PolyData(exp_cells)
    neighs = pv.PolyData(neigh_cells)
    
    plotter = pv.BackgroundPlotter(title='exp_cells', 
                                   window_size=[1024, 768])
    
    plotter.add_mesh(neighs, show_edges=True,
                     scalars=neighs.points[:,2],
                     point_size=10,
                     render_points_as_spheres=True,
                     cmap='inferno',
                     clim=[min(neighs.points[:,2]), wafer_thickness])
    plotter.add_scalar_bar(title='z height',height=0.08,width=0.4,
                           position_x=0.01,position_y=0.01)
    
#    plotter.add_mesh(cells, show_edges=True,
#                     scalars=exp_states,#cells.points[:,2],
#                     point_size=8,
#                     render_points_as_spheres=True,
#                     cmap='inferno',
#                     clim=[0,1])#[min(cells.points[:,2]), wafer_thickness])
#    plotter.add_scalar_bar(title='z height',height=0.08,width=0.4,
#                           position_x=0.01,position_y=0.1)

    plotter.add_text(step)



######

pts = etch_grid[loop_steps[-1]][0]
states = etch_grid[loop_steps[-1]][1]
iden = etch_grid[loop_steps[-1]][2]

exp_cell_idx = np.where(iden == 1)[0]
      
exp_cells,exp_states = pts[exp_cell_idx],states[exp_cell_idx]



exposed_obj = pv.read(vtk_save_exp_obj)#pv.PolyData(exp_cells) 
smooth = exposed_obj.smooth(n_iter=100)
plotter = pv.BackgroundPlotter(title=loop_steps[-1], 
                               window_size=[1024, 768])
plotter.add_mesh(exposed_obj, show_edges=True,
                 scalars=exposed_obj.points[:,2],
                 point_size=8,
                 render_points_as_spheres=True,
                 cmap='inferno',
                 clim=[min(exposed_obj.points[:,2]), wafer_thickness])
plotter.add_scalar_bar(title='z_height',height=0.08,width=0.4,
                       position_x=0.01,position_y=0.1)
      

neigh_obj = pv.read(vtk_save_neigh_obj)#pv.PolyData(exp_cells) 
smooth = neigh_obj.smooth(n_iter=1000)
plotter = pv.BackgroundPlotter(title=loop_steps[-1], 
                               window_size=[1024, 768])
plotter.add_mesh(neigh_obj, show_edges=True,
                 scalars=smooth.points[:,2],
                 point_size=8,
                 render_points_as_spheres=True,
                 cmap='inferno',
                 clim=[min(smooth.points[:,2]), wafer_thickness])
plotter.add_scalar_bar(title='z_height',height=0.08,width=0.4,
                       position_x=0.01,position_y=0.1)
   

   
cells = np.asarray(exposed_cells.points)
#obj = make_grid([cells],cell_size)

obj = pv.read('C:/Users/nicas/Documents/E241-MicroNanoFab/codes/etch_model_version5_1/neigh_obj.vtk')
obj = make_grid([np.array(obj.points)],cell_size)
slices = obj.slice(normal=[1,1,0])
plotter = pv.BackgroundPlotter(window_size=[1024, 768])
plotter.add_mesh(obj, show_edges=False,
                         scalars=obj.points[:,2],
                         cmap='inferno',
                         clim=[min(obj.points[:,2]), 
                               wafer_thickness])

x,z = cross_section_slice(cells,cell_size,p1=(-200,-200),p2=(200,200))
plt.plot(x,z)
