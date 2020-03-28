import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy.spatial import distance as dist
from matplotlib.path import Path
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from shapely.ops import cascaded_union
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
mpl.style.use('default')




"""
Written by Nicolas Castano

Model continuous etching into silicon wafer on the PT-DSE tool in the SNF 
based known etch rates.
"""   

def horiz_etch(cont,horiz_rate,t_step,norm_span,sm_window):
    # method to apply normal step to a contour
    out_cont = np.zeros_like(cont)
    for p, point in enumerate(cont):
#        x_q = point[0]  # if you want to plot a quiver for normal lines
#        y_q = point[1]
        # calculate normal to point
        # enable looping index to beginning
        if p + norm_span > len(cont)-1:
            dummy_p = 0
        else:
            dummy_p = p
        p_1 = cont[dummy_p-norm_span]
        p_2 = cont[dummy_p+norm_span]
        tan_vec = np.array([[p_2[0]-p_1[0]],
                         [p_2[1]-p_1[1]]])
        norm_vec = np.matmul(rot90_mat,tan_vec)
        unit_norm = norm_vec/np.linalg.norm(norm_vec)
        
#            if t==30:
#                ax.quiver(x_q,y_q,unit_norm[0],unit_norm[1],width=0.002)
#            ax.plot(cont[:,0],cont[:,1],'o')
       
        # calculate new point
        new_pt = point + horiz_rate*t_step*np.reshape(unit_norm,(1,2))
        out_cont[p,:] = new_pt
    
    # fprce last point to be on top of first in contour
    out_cont[-1,0] = out_cont[0,0]
    out_cont[-1,1] = out_cont[0,1]
    # smooth with spline
    tck, u = splprep(out_cont.T, u=None, s=0, per=1) 
    u_new = np.linspace(u.min(), u.max(), len(cont))
    x_spline, y_spline = splev(u_new, tck, der=0)
    out_cont = np.hstack((np.reshape(np.array(x_spline),[len(x_spline),1]),
                          np.reshape(np.array(y_spline),[len(y_spline),1])))        
    
    return out_cont

#def animate(index):
#    zi = ml.griddata(x, y, zlist[index], xi, yi, interp='linear')
#    ax.clear()
#    ax.contourf(xi, yi, zi, **kw)
#    ax.set_title('%03d'%(index))



    
C4F8 = 100  # sccm
SF6 = 300  # sccm
bias = 10  # volts
time = 600  # seconds
opening = 100  # um

plt.close('all')

# load mask
im_dir = 'C:/Users/nicas/Documents/E241-MicroNanoFab/masks/'
im_file = 'python_model_fil_sq.png'
im_path = im_dir + im_file
curr_im = cv2.imread(im_path, cv2.IMREAD_ANYDEPTH)   
curr_im = cv2.GaussianBlur(curr_im,(3,3),0)


rgb_im = cv2.cvtColor(curr_im, cv2.COLOR_GRAY2RGB)
     
cont_im, conts, hier = cv2.findContours(curr_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)    
conts_im = cv2.drawContours(rgb_im, conts, -1, (0,255,0),3)


dummy_i = im_file.find('.png')
out_file = im_dir + im_file[:dummy_i] + '_out' + im_file[dummy_i:]
cv2.imwrite(out_file, conts_im)


t_start = 0
t_end = 600  # seconds
t_step = 5
h = curr_im.shape[0]
w = curr_im.shape[1]
n_points = 600
contour_read_step = 5
topo_im = np.zeros_like(curr_im)
norm_span = 3
window_len = 17
rot90_mat = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)],
                      [np.sin(np.pi/2), np.cos(np.pi/2)]])
vert_rate = 287/600  # um/s

horiz_rate = 77/600  # um/s
pixel_um_conv = 251/90.4672  # px/um
cmap = 'gnuplot'  # 'inferno' 'viridis'  # 'hot'
vmin = -290  # expected range of depth for color bar (min)
vmax = 0
# for plotting srface plot
rstride = 2
cstride = 2


x_axis = np.linspace(0,w/pixel_um_conv,n_points)
y_axis = np.linspace(0,h/pixel_um_conv,n_points)
xv,yv = np.meshgrid(x_axis,y_axis)
x_points = np.ravel(xv)
x_points = x_points.reshape((len(x_points),1))
y_points = np.ravel(yv)
y_points = y_points.reshape((len(y_points),1))
grid_point_pairs = np.hstack((x_points,y_points))



# get points for each contour
# tracking paths and polygons 
conts_paths = {}
conts_polys = {}
topo_data = {}
unit_norm_vectors = {}


#fig, ax = plt.subplots()
for c, cont in enumerate(conts):
    x = []
    y = []
    for p, point in enumerate(cont):
        if p%contour_read_step == 0:
            x.append(point[0][0]/pixel_um_conv)
            y.append(point[0][1]/pixel_um_conv)    
#            plt.scatter(point[0][0],point[0][1],3,'k')
#            plt.text(point[0][0],point[0][1],str(p))
    
    # force last point to be on top of first point
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


    temp_poly = Polygon(points)
    temp_path = Path(temp_poly.exterior,closed=True)    
    conts_paths[c] = temp_path  # path object nice for the contains_point attribute
    conts_polys[c] = temp_poly
    
    unit_norm_vectors[c] = np.zeros_like(conts_paths[c].vertices)
    topo_data[c] = np.zeros((grid_point_pairs.shape[0],1))  # each point will have a depth
    

#    patch = patches.PathPatch(temp_path, facecolor='orange', lw=2)
#    ax.add_patch(patch)
#ax.autoscale_view()
#plt.show()
    
x = grid_point_pairs[:,0].reshape(xv.shape)
y = grid_point_pairs[:,1].reshape(yv.shape)

fig2, ax2 = plt.subplots(figsize=(8,7))

dummy_cont_count = len(conts_paths)

topo = []

# solve the etching of the mask
for i_t,t in enumerate(range(t_start, t_end, t_step)):
#    vert_rate = (10+2/600*t)/60  # (10 + 0.0000056969697*t**2)/60
#    horiz_rate = (3-(-0.1318335/0.04394449)*(1-np.exp(-0.04394449*t)))/60
    print('solving time: ', t)
    z_mask = np.zeros_like(xv)  # 1 if etch back at node, 0 if not
    topo.append(np.zeros_like(xv))
    cont_loop = True
    overlap = False
    cummul_paths = {}
    c = 0
    
    # determine the overlapping points in adjacents mask openings
    # 0 for no mask opening, 1 for opening 
    # solving for surface level contour
    while cont_loop == True:
#    for c in cont_arrays:
        print('  checking for OVERLAP with contour: ', c)
        # determine if contours overlap      
        other_conts = list(range(dummy_cont_count))
        other_conts.pop(other_conts.index(c))
        for oc in other_conts:
            for pt in conts_paths[oc].vertices:
                if conts_paths[c].contains_point(pt):
                    overlap = True
                    break
            if overlap == True: break
        
        
#        # stack all contours and convert to binary mask
#        
#        inside = conts_paths[c].contains_points(grid_point_pairs)
#        z_mask += inside.astype(int).reshape(xv.shape)
#        z_mask[z_mask>0] = 1
        
        



            

        
        # if one overlaps assume all are overlapping
        if overlap == True: 
            print('     overlap detected')
            #combine contours
            polys = [conts_polys[poly] for poly in list(conts_polys.keys())]
            new_cont = cascaded_union([poly if poly.is_valid 
                                       else poly.buffer(0) for poly in polys])
    
            # smooth with spline
            try:
                x_temp,y_temp = new_cont.exterior.xy
                x_temp[-1] = x_temp[0]
                y_temp[-1] = y_temp[0]
                temp_cont = np.hstack((np.reshape(np.array(x_temp),[len(x_temp),1]),
                                       np.reshape(np.array(y_temp),[len(y_temp),1])))
                tck, u = splprep(temp_cont.T, u=None, s=13, per=1) 
                u_new = np.linspace(u.min(), u.max(), len(cont))
                x_spline, y_spline = splev(u_new, tck, der=0)
                points = np.hstack((np.reshape(np.array(x_temp),[len(x_temp),1]),
                                    np.reshape(np.array(y_temp),[len(y_temp),1])))
                cummul_paths[c] = Path(points,closed=True)
                
                # false to exit while loop
                cont_loop = False
            except:
                overlap = False
                
            
            
        if overlap == False:
            # check if points are inside contour (removed from mask)
            cummul_paths[c] = conts_paths[c]
            c += 1
            if c == dummy_cont_count: cont_loop = False
    

    # adjust contours, ignoring the overlap to get true vertical etch
    # stack all contours and convert to binary mask
    for c in conts_paths:
        print('  solving VERT etch in contour: ', c)
        new_cont_points = horiz_etch(conts_paths[c].vertices,horiz_rate,
                                     t_step,norm_span,window_len)

        conts_paths[c] = Path(new_cont_points,closed=True)
        conts_polys[c] = Polygon(new_cont_points)

        inside = conts_paths[c].contains_points(grid_point_pairs)
        z_mask += inside.astype(int).reshape(xv.shape)
    z_mask[z_mask>0] = 1
    # update topography using the z_mask
    z_step = z_mask * (vert_rate*t_step)
    # etch back; try to reference last time step, except its the first time step
    try:
        topo[i_t] =  topo[i_t-1] - z_step
    except:
        topo[i_t] -= z_step
        
        
#    # dummy plot
    ax2.plot(t,vert_rate/horiz_rate,'k')
#    if i_t>400 and c==3:
#        patch = patches.PathPatch(conts_paths[c], fill=False, lw=2)
#        ax2.add_patch(patch) 
#        contourplot = plt.contourf(x, y, z_mask, 100, cmap=cmap,vmin=0, vmax=2)
#        ax2.autoscale()
#        ax2, _ = mpl.colorbar.make_axes(plt.gca())
#        cbar = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
#                                 norm=mpl.colors.Normalize(vmin=0, vmax=2),
#                                 label=' etch depth [um]')
#        cbar.set_clim(0, 2)

        
        
    fig_2d_cont, ax1_2d_cont = plt.subplots(figsize=(13,12))

    # now solve the horizontal step in the combined contour
    for c in cummul_paths:
        print('  solving HORIZ etch in cummulative contour: ', c)
            
        curr_path = cummul_paths[c]        
        try:
            updated_cont = horiz_etch(curr_path.vertices,horiz_rate,
                                      t_step,norm_span,window_len)
        except:
            pass
        dummy_cont_path = Path(updated_cont,closed=True)
        patch = patches.PathPatch(dummy_cont_path, fill=False, lw=2)
        ax1_2d_cont.add_patch(patch)   
        ax1_2d_cont.plot(dummy_cont_path.vertices[:,0],
                 dummy_cont_path.vertices[:,1],'k')
    
    
    # plot 2d contour
    contourplot = plt.contourf(x, y, topo[i_t], 500, cmap=cmap,vmin=vmin, vmax=vmax)
    title_str = 't = %s s' % str(t)
    plt.title(title_str)
    ax1_2d_cont, _ = mpl.colorbar.make_axes(plt.gca())
    cbar = mpl.colorbar.ColorbarBase(ax1_2d_cont, cmap=cmap,
                                     norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                                     label=' etch depth [um]')
    cbar.set_clim(vmin, vmax)    
    ax1_2d_cont.autoscale()    
    out_fig = 'C:/Users/nicas/Documents/E241-MicroNanoFab/codes/comb_contours/' + \
    str(t) + '.png'
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
        
    
    
    # plot 3d surface
    fig_3d_surf = plt.figure(figsize=(24,10))
#    ax2_3d_surf = fig_3d_surf.gca(projection='3d')
    ax2_3d_surf = fig_3d_surf.add_subplot(111, projection='3d')

#    fig_3d_surf = plt.figure(figsize=(20,11))
#    ax2_3d_surf = fig_3d_surf.add_subplot(1,2,1,projection='3d')
    surf = ax2_3d_surf.plot_surface(x, y, topo[i_t-1], rstride=rstride, 
                                    cstride=cstride, 
                                    cmap=cmap,vmin=vmin, vmax=vmax,
                                    linewidth=0, antialiased=False)
    ax2_3d_surf.set_zlim(vmin, vmax)
    ax2_3d_surf.view_init(65, -60)
    title_str = 't = %s s' % str(t)
    plt.title(title_str)
    # add a color bar which maps values to colors.
    ax2_3d_surf, _ = mpl.colorbar.make_axes(plt.gca())
    cbar = mpl.colorbar.ColorbarBase(ax2_3d_surf, cmap=cmap,
                                     norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                                     label=' etch depth [um]')
    cbar.set_clim(vmin, vmax)    
    ax2_3d_surf.autoscale()    
#    fig_3d_surf.colorbar(surf, shrink=0.5, aspect=5)
    out_fig = 'C:/Users/nicas/Documents/E241-MicroNanoFab/codes/comb_contours_3d/' + \
    str(t) + '.png'
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()



fig3, ax3 = plt.subplots(figsize=(8,7))
for i,_ in enumerate(x):
    line_x = np.sqrt(x[i,i]**2 + y[i,i]**2)
    ax3.scatter(line_x,topo[i_t-1][i,i])
    #    ax2.set_title(title_str)
            
    #    ax2.add_patch(patch_dummy)
    ax3.autoscale()
plt.show()

      
fig3, ax3 = plt.subplots(figsize=(8,7))              
dummy_i = int(n_points/2)
plt.plot(x[dummy_i,:],topo[i_t-1][dummy_i,:])
#    ax2.set_title(title_str)
        
#    ax2.add_patch(patch_dummy)
plt.show()
    
