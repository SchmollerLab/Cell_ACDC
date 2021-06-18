import numpy as np
import cv2
import skimage.measure
import skimage.morphology
import skimage.exposure
import skimage.draw
import skimage.registration
import skimage.color
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, PathPatch, Path

# Custom modules
from MyWidgets import Slider, Button, RadioButtons
import apps

def align_frames_3D(data, slices=None, register=True, user_shifts=None):
    registered_shifts = np.zeros((len(data),2), int)
    data_aligned = np.copy(data)
    for frame_i, frame_V in enumerate(data):
        slice = slices[frame_i]
        if frame_i != 0:  # skip first frame
            curr_frame_img = frame_V[slice]
            prev_frame_img = data_aligned[frame_i-1, slice] #previously aligned frame, slice
            if register==True:
                shifts = skimage.registration.phase_cross_correlation(
                    prev_frame_img, curr_frame_img
                    )[0]
            else:
                shifts = user_shifts[frame_i]
            shifts = shifts.astype(int)
            aligned_frame_V = np.copy(frame_V)
            aligned_frame_V = np.roll(aligned_frame_V, tuple(shifts), axis=(1,2))
            data_aligned[frame_i] = aligned_frame_V
            registered_shifts[frame_i] = shifts
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(z_proj_max(frame_V))
            # ax[1].imshow(z_proj_max(aligned_frame_V))
            # plt.show()
    return data_aligned, registered_shifts


def align_frames_2D(data, slices=None, register=True, user_shifts=None):
    registered_shifts = np.zeros((len(data),2), int)
    data_aligned = np.copy(data)
    for frame_i, frame_V in enumerate(data):
        if frame_i != 0:  # skip first frame
            curr_frame_img = frame_V
            prev_frame_img = data_aligned[frame_i-1] #previously aligned frame, slice
            if register==True:
                shifts = skimage.registration.phase_cross_correlation(
                    prev_frame_img, curr_frame_img
                    )[0]
            else:
                shifts = user_shifts[frame_i]
            shifts = shifts.astype(int)
            aligned_frame_V = np.copy(frame_V)
            aligned_frame_V = np.roll(aligned_frame_V, tuple(shifts), axis=(0,1))
            data_aligned[frame_i] = aligned_frame_V
            registered_shifts[frame_i] = shifts
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(z_proj_max(frame_V))
            # ax[1].imshow(z_proj_max(aligned_frame_V))
            # plt.show()
    return data_aligned, registered_shifts

class my_paint_app:
    def __init__(self, label_img, ID, rp, eps_percent=0.01, del_small_obj=False,
                 overlay_img=None):
        # matplolib dark mode
        plt.style.use('dark_background')
        plt.rc('axes', edgecolor='0.1')

        """Initialize attributes"""
        self.cancel = False
        self.ID_bud = label_img.max() + 1
        self.ID_moth = ID
        self.label_img = label_img
        self.coords_delete = []
        self.overlay_img = skimage.exposure.equalize_adapthist(overlay_img)
        self.num_cells = 1
        """Build image containing only selected ID obj"""
        only_ID_img = np.zeros_like(label_img)
        only_ID_img[label_img == ID] = ID
        all_IDs = [obj.label for obj in rp]
        obj_rp = rp[all_IDs.index(ID)]
        min_row, min_col, max_row, max_col = obj_rp.bbox
        obj_bbox_h = max_row - min_row
        obj_bbox_w = max_col - min_col
        side_len = max([obj_bbox_h, obj_bbox_w])
        obj_bbox_cy = min_row + obj_bbox_h/2
        obj_bbox_cx = min_col + obj_bbox_w/2
        obj_bottom = int(obj_bbox_cy - side_len/2)
        obj_left = int(obj_bbox_cx - side_len/2)
        obj_top = obj_bottom + side_len
        obj_right = obj_left + side_len
        self.bw = 10
        self.xlims = (obj_left-self.bw, obj_right+self.bw)
        self.ylims = (obj_top+self.bw, obj_bottom-self.bw)
        self.only_ID_img = only_ID_img
        self.sep_bud_label = only_ID_img.copy()
        self.eraser_mask = np.zeros(self.label_img.shape, bool)
        self.small_obj_mask = np.zeros(only_ID_img.shape, bool)

        """generate image plot and connect to events"""
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot()
        self.fig.subplots_adjust(bottom=0.25)
        (self.ax).imshow(self.only_ID_img)
        (self.ax).set_xlim(*self.xlims)
        (self.ax).set_ylim(self.ylims)
        (self.ax).axis('off')
        (self.fig).suptitle('Draw a curve with the right button to separate cell.\n'
                            'Delete object with mouse wheel button\n'
                            'Erase with mouse left button', y=0.95)

        """Find convexity defects"""
        try:
            cnt, defects = self.convexity_defects(
                                              self.only_ID_img.astype(np.uint8),
                                              eps_percent)
        except:
            defects = None
        if defects is not None:
            defects_points = [0]*len(defects)
            for i, defect in enumerate(defects):
                s,e,f,d = defect[0]
                x,y = tuple(cnt[f][0])
                defects_points[i] = (y,x)
                self.ax.plot(x,y,'r.')

        """Embed plt window into a tkinter window"""
        sub_win = apps.embed_tk('Mother-bud zoom', [1024,768,400,150], self.fig)


        """Create buttons"""
        self.ax_ok_B = self.fig.add_subplot(position=[0.2, 0.2, 0.1, 0.03])
        self.ax_overlay_B = self.fig.add_subplot(position=[0.8, 0.2, 0.1, 0.03])
        self.alpha_overlay_sl_ax = self.fig.add_subplot(
                                                 position=[0.7, 0.2, 0.1, 0.03])
        self.brightness_overlay_sl_ax = self.fig.add_subplot(
                                                 position=[0.6, 0.2, 0.1, 0.03])
        self.ok_B = Button(self.ax_ok_B, 'Happy\nwith that', canvas=sub_win.canvas,
                            color='0.1', hovercolor='0.25', presscolor='0.35')
        self.overlay_B = Button(self.ax_overlay_B, 'Overlay',
                            canvas=sub_win.canvas,
                            color='0.1', hovercolor='0.25', presscolor='0.35')
        self.alpha_overlay_sl = Slider(self.alpha_overlay_sl_ax,
                           'alpha', -0.1, 1.1,
                            canvas=sub_win.canvas,
                            valinit=0.3,
                            valstep=0.01,
                            color='0.2',
                            init_val_line_color='0.25',
                            valfmt='%1.2f',
                            orientation='vertical')
        self.brightness_overlay_sl = Slider(self.brightness_overlay_sl_ax,
                           'brightness', 0, 2,
                            canvas=sub_win.canvas,
                            valinit=1,
                            valstep=0.01,
                            color='0.2',
                            init_val_line_color='0.25',
                            valfmt='%1.2f',
                            orientation='vertical')
        """Connect to events"""
        (sub_win.canvas).mpl_connect('button_press_event', self.mouse_down)
        (sub_win.canvas).mpl_connect('button_release_event', self.mouse_up)
        self.cid_brush_circle = (sub_win.canvas).mpl_connect(
                                                    'motion_notify_event',
                                                    self.draw_brush_circle)
        (sub_win.canvas).mpl_connect('key_press_event', self.key_down)
        (sub_win.canvas).mpl_connect('resize_event', self.resize)
        (sub_win.root).protocol("WM_DELETE_WINDOW", self.abort_exec)
        self.overlay_B.on_clicked(self.toggle_overlay)
        self.ok_B.on_clicked(self.ok)
        self.alpha_overlay_sl.on_changed(self.update_img)
        self.brightness_overlay_sl.on_changed(self.update_img)
        self.sub_win = sub_win
        self.clicks_count = 0
        self.brush_size = 2
        self.eraser_on = True
        self.overlay_on = False
        self.set_labRGB_colors()
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        sub_win.root.mainloop()

    def toggle_overlay(self, event):
        self.overlay_on = not self.overlay_on
        if self.overlay_on:
            self.alpha_overlay_sl_ax.set_visible(True)
            self.brightness_overlay_sl_ax.set_visible(True)
        else:
            self.alpha_overlay_sl_ax.set_visible(False)
            self.brightness_overlay_sl_ax.set_visible(False)
        self.update_img(None)

    def set_labRGB_colors(self):
        # Generate a colormap as sparse as possible given the max ID.
        gradient = np.linspace(255, 0, self.num_cells, dtype=int)
        labelRGB_colors = np.asarray([plt.cm.viridis(i) for i in gradient])
        self.labRGB_colors = labelRGB_colors

    def key_down(self, event):
        key = event.key
        if key == 'enter':
            self.ok(None)
        elif key == 'ctrl+z':
            self.undo(None)
        elif key == 'up':
            self.brush_size += 1
            self.draw_brush_circle(event)
        elif key == 'down':
            self.brush_size -= 1
            self.draw_brush_circle(event)
        elif key == 'x':
            # Switch eraser mode on or off
            self.eraser_on = not self.eraser_on
            self.draw_brush_circle(event)

    def resize(self, event):
        # [left, bottom, width, height]
        (self.ax_left, self.ax_bottom,
        self.ax_right, self.ax_top) = self.ax.get_position().get_points().flatten()
        B_h = 0.08
        B_w = 0.1
        self.ax_ok_B.set_position([self.ax_right-B_w, self.ax_bottom-B_h-0.01,
                                   B_w, B_h])
        self.ax_overlay_B.set_position([self.ax_left, self.ax_bottom-B_h-0.01,
                                   B_w*2, B_h])
        self.alpha_overlay_sl_ax.set_position([self.ax_right+0.05,
                                               self.ax_bottom,
                                               B_w/3,
                                               self.ax_top-self.ax_bottom])
        self.brightness_overlay_sl_ax.set_position([
                                               self.ax_right+0.05+B_w/3+0.05,
                                               self.ax_bottom,
                                               B_w/3,
                                               self.ax_top-self.ax_bottom])
        if self.overlay_img is None:
            self.ax_overlay_B.set_visible(False)
        self.alpha_overlay_sl_ax.set_visible(False)
        self.brightness_overlay_sl_ax.set_visible(False)

    def update_img(self, event):
        lab = self.sep_bud_label.copy()
        for y, x in self.coords_delete:
            del_ID = self.sep_bud_label[y, x]
            lab[lab == del_ID] = 0
        rp = skimage.measure.regionprops(lab)
        num_cells = len(rp)
        if self.num_cells != num_cells:
            self.set_labRGB_colors()
        if not self.overlay_on:
            img = lab
        else:
            brightness = self.brightness_overlay_sl.val
            img = skimage.color.label2rgb(
                                lab,image=self.overlay_img*brightness,
                                bg_label=0,
                                bg_color=(0.1,0.1,0.1),
                                colors=self.labRGB_colors,
                                alpha=self.alpha_overlay_sl.val
                                )
            img = np.clip(img, 0, 1)
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.set_xlim(*self.xlims)
        self.ax.set_ylim(*self.ylims)
        self.ax.axis('off')
        for t in self.ax.texts:
            t.set_visible(False)
        for obj in rp:
            y, x = obj.centroid
            txt = f'{obj.label}'
            self.ax.text(
                    int(x), int(y), txt, fontsize=18,
                    fontweight='semibold', horizontalalignment='center',
                    verticalalignment='center', color='k', alpha=1)
        (self.sub_win.canvas).draw_idle()

    def mouse_down(self, event):
        if event.inaxes == self.ax and event.button == 3:
            x = int(event.xdata)
            y = int(event.ydata)
            if self.clicks_count == 0:
                self.x0 = x
                self.y0 = y
                self.cid_line = (self.sub_win.canvas).mpl_connect(
                                                         'motion_notify_event',
                                                                self.draw_line)
                self.pltLine = Line2D([self.x0, self.x0], [self.y0, self.y0])
                self.clicks_count = 1
            elif self.clicks_count == 1:
                self.x1 = x
                self.y1 = y
                (self.sub_win.canvas).mpl_disconnect(self.cid_line)
                self.cid_bezier = (self.sub_win.canvas).mpl_connect(
                                                         'motion_notify_event',
                                                              self.draw_bezier)
                self.clicks_count = 2
            elif self.clicks_count == 2:
                self.x2 = x
                self.y2 = y
                (self.sub_win.canvas).mpl_disconnect(self.cid_bezier)
                self.separate_cb()
                self.clicks_count = 0

        elif event.inaxes == self.ax and event.button == 2:
            xp = int(event.xdata)
            yp = int(event.ydata)
            self.coords_delete.append((yp, xp))
            self.update_img(None)

        elif event.inaxes == self.ax and event.button == 1:
            (self.sub_win.canvas).mpl_disconnect(self.cid_brush_circle)
            self.xb, self.yb = self.ax_transData_and_coerce(self.ax, event.x,
                                                                     event.y,
                                                        self.label_img.shape)
            self.apply_brush(event)
            self.cid_brush = (self.sub_win.canvas).mpl_connect(
                                                     'motion_notify_event',
                                                          self.apply_brush)

    def get_poly_brush(self, yxc1, yxc2, r):
        # see https://en.wikipedia.org/wiki/Tangent_lines_to_circles
        R = r
        y1, x1 = yxc1
        y2, x2 = yxc2
        arcsin_den = np.sqrt((x2-x1)**2+(y2-y1)**2)
        arctan_den = (x2-x1)
        if arcsin_den!=0 and arctan_den!=0:
            beta = np.arcsin((R-r)/arcsin_den)
            gamma = -np.arctan((y2-y1)/arctan_den)
            alpha = gamma-beta
            x3 = x1 + r*np.sin(alpha)
            y3 = y1 + r*np.cos(alpha)
            x4 = x2 + R*np.sin(alpha)
            y4 = y2 + R*np.cos(alpha)

            alpha = gamma+beta
            x5 = x1 - r*np.sin(alpha)
            y5 = y1 - r*np.cos(alpha)
            x6 = x2 - R*np.sin(alpha)
            y6 = y2 - R*np.cos(alpha)

            rr_poly, cc_poly = skimage.draw.polygon([y3, y4, y6, y5],
                                                    [x3, x4, x6, x5])
        else:
            rr_poly, cc_poly = [], []
        return rr_poly, cc_poly

    def apply_brush(self, event):
        if event.button == 1:
            x, y = self.ax_transData_and_coerce(self.ax, event.x, event.y,
                                                        self.label_img.shape)

            rr, cc = skimage.draw.disk((y, x), radius=self.brush_size,
                                               shape=self.label_img.shape)
            rr_poly, cc_poly = self.get_poly_brush((self.yb, self.xb), (y, x),
                                                    self.brush_size)
            self.xb, self.yb = x, y
            if self.eraser_on:
                self.eraser_mask[rr, cc] = True
                self.eraser_mask[rr_poly, cc_poly] = True
                self.sep_bud_label[self.eraser_mask] = 0
            else:
                self.sep_bud_label[rr, cc] = self.ID_moth
                self.sep_bud_label[rr_poly, cc_poly] = self.ID_moth
                self.eraser_mask[rr, cc] = False
                self.eraser_mask[rr_poly, cc_poly] = False
            self.update_img(None)
            c = 'r' if self.eraser_on else 'g'
            self.brush_circle = matplotlib.patches.Circle((x, y),
                                    radius=self.brush_size,
                                    fill=False,
                                    color=c, lw=2)
            (self.ax).add_patch(self.brush_circle)
            (self.sub_win.canvas).draw_idle()


    def draw_line(self, event):
        if event.inaxes == self.ax:
            self.yd = int(event.ydata)
            self.xd = int(event.xdata)
            self.pltLine.set_visible(False)
            self.pltLine = Line2D([self.x0, self.xd], [self.y0, self.yd],
                                   color='r', ls='--')
            self.ax.add_line(self.pltLine)
            (self.sub_win.canvas).draw_idle()

    def draw_bezier(self, event):
        self.xd, self.yd = self.ax_transData_and_coerce(self.ax, event.x,
                                                                 event.y,
                                                    self.label_img.shape)
        try:
            self.plt_bezier.set_visible(False)
        except:
            pass
        p0 = (self.x0, self.y0)
        p1 = (self.xd, self.yd)
        p2 = (self.x1, self.y1)
        self.plt_bezier = PathPatch(
                                 Path([p0, p1, p2],
                                      [Path.MOVETO,
                                       Path.CURVE3,
                                       Path.CURVE3]),
                                     fc="none", transform=self.ax.transData,
                                     color='r')
        self.ax.add_patch(self.plt_bezier)
        (self.sub_win.canvas).draw_idle()

    def ax_transData_and_coerce(self, ax, event_x, event_y, img_shape,
                                return_int=True):
        x, y = ax.transData.inverted().transform((event_x, event_y))
        ymax, xmax = img_shape
        xmin, ymin = 0, 0
        if x < xmin:
            x_coerced = 0
        elif x > xmax:
            x_coerced = xmax-1
        else:
            x_coerced = int(round(x)) if return_int else x
        if y < ymin:
            y_coerced = 0
        elif y > ymax:
            y_coerced = ymax-1
        else:
            y_coerced = int(round(y)) if return_int else y
        return x_coerced, y_coerced



    def nearest_nonzero(self, a, y, x):
        r, c = np.nonzero(a)
        dist = ((r - y)**2 + (c - x)**2)
        min_idx = dist.argmin()
        return a[r[min_idx], c[min_idx]]

    def separate_cb(self):
        c0, r0 = (self.x0, self.y0)
        c1, r1 = (self.x2, self.y2)
        c2, r2 = (self.x1, self.y1)
        rr, cc = skimage.draw.bezier_curve(r0, c0, r1, c1, r2, c2, 1)
        sep_bud_img = np.copy(self.sep_bud_label)
        sep_bud_img[rr, cc] = 0
        self.sep_bud_img = sep_bud_img
        sep_bud_label_0 = skimage.measure.label(self.sep_bud_img, connectivity=1)
        sep_bud_label = skimage.morphology.remove_small_objects(
                                             sep_bud_label_0,
                                             min_size=20,
                                             connectivity=2)
        small_obj_mask = np.logical_xor(sep_bud_label_0>0,
                                        sep_bud_label>0)
        self.small_obj_mask = np.logical_or(small_obj_mask,
                                            self.small_obj_mask)
        rp_sep = skimage.measure.regionprops(sep_bud_label)
        IDs = [obj.label for obj in rp_sep]
        max_ID = self.ID_bud+len(IDs)
        sep_bud_label[sep_bud_label>0] = sep_bud_label[sep_bud_label>0]+max_ID
        rp_sep = skimage.measure.regionprops(sep_bud_label)
        IDs = [obj.label for obj in rp_sep]
        areas = [obj.area for obj in rp_sep]
        curr_ID_bud = IDs[areas.index(min(areas))]
        curr_ID_moth = IDs[areas.index(max(areas))]
        sep_bud_label[sep_bud_label==curr_ID_moth] = self.ID_moth
        # sep_bud_label = np.zeros_like(sep_bud_label)
        sep_bud_label[sep_bud_label==curr_ID_bud] = self.ID_bud+len(IDs)-2
        temp_sep_bud_lab = sep_bud_label.copy()
        self.rr = []
        self.cc = []
        self.val = []
        for r, c in zip(rr, cc):
            if self.only_ID_img[r, c] != 0:
                ID = self.nearest_nonzero(sep_bud_label, r, c)
                temp_sep_bud_lab[r,c] = ID
                self.rr.append(r)
                self.cc.append(c)
                self.val.append(ID)
        self.sep_bud_label = temp_sep_bud_lab
        self.update_img(None)

    def mouse_up(self, event):
        try:
            (self.sub_win.canvas).mpl_disconnect(self.cid_brush)
            self.cid_brush_circle = (self.sub_win.canvas).mpl_connect(
                                                        'motion_notify_event',
                                                        self.draw_brush_circle)
        except:
            pass

    def draw_brush_circle(self, event):
        if event.inaxes == self.ax:
            x, y = self.ax_transData_and_coerce(self.ax, event.x, event.y,
                                                        self.label_img.shape)
            try:
                self.brush_circle.set_visible(False)
            except:
                pass
            c = 'r' if self.eraser_on else 'g'
            self.brush_circle = matplotlib.patches.Circle((x, y),
                                    radius=self.brush_size,
                                    fill=False,
                                    color=c, lw=2)
            self.ax.add_patch(self.brush_circle)
            (self.sub_win.canvas).draw_idle()

    def convexity_defects(self, img, eps_percent):
        contours, hierarchy = cv2.findContours(img,2,1)
        cnt = contours[0]
        cnt = cv2.approxPolyDP(cnt,eps_percent*cv2.arcLength(cnt,True),True) # see https://www.programcreek.com/python/example/89457/cv22.convexityDefects
        hull = cv2.convexHull(cnt,returnPoints = False) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        defects = cv2.convexityDefects(cnt,hull) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        return cnt, defects

    def undo(self, event):
        self.coords_delete = []
        sep_bud_img = np.copy(self.only_ID_img)
        self.sep_bud_img = sep_bud_img
        self.sep_bud_label = np.copy(self.only_ID_img)
        self.small_obj_mask = np.zeros(self.only_ID_img.shape, bool)
        self.eraser_mask = np.zeros(self.label_img.shape, bool)
        self.overlay_on = False
        rp = skimage.measure.regionprops(sep_bud_img)
        self.ax.clear()
        self.ax.imshow(self.sep_bud_img)
        (self.ax).set_xlim(*self.xlims)
        (self.ax).set_ylim(*self.ylims)
        text_label_centroid(rp, self.ax, 18, 'semibold', 'center',
                            'center', None, display_ccStage=False,
                            color='k', clear=True)
        self.ax.axis('off')
        (self.sub_win.canvas).draw_idle()

    def ok(self, event):
        # plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()

    def abort_exec(self):
        self.cancel = True
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()
