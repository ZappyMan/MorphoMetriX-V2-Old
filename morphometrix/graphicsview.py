from PySide6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsTextItem, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsPolygonItem
from PySide6 import QtGui, QtCore
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtCore import Qt, QLineF
from scipy.linalg import pascal
from scipy.sparse import diags
from scipy.optimize import root_scalar
from itertools import cycle, islice
import numpy as np
import sys, os, types

# ------------------------------
#   MorphoMetrix GraphicsView Class - Developed By:
#       Walter Torres
#       Kevin Bierlich
#       Clara Bird
#       Elliott Chimienti
# ------------------------------

consts = types.SimpleNamespace()
# State case constants
consts.LENGTH = 1
consts.AREA = 2
consts.ANGLE = 3
consts.WIDTH = 4

# Object type constants
consts.LINEITEM = 1
consts.PATHITEM = 2
consts.ELLIPSEITEM = 3
consts.FONTITEM = 4
consts.POLYGONITEM = 5

# Side bias constants
consts.SIDE_A = 0
consts.SIDE_B = 1

class imwin(QGraphicsView):
    def __init__(self, parent=None):
        super(imwin, self).__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.picked_color = QtGui.QColor("red")     # Default width ellipse color
        self.slider_pos = 10                        # Used by Ellipse Class
        self.opacity_pos = 10                       # Used by Ellipse Class
        self.numwidths = None                       # User defined for width measurements
        self.measurement_stack = []                 # Initialize Empty Stack (FIFO)
        self.measuring_state = None                 # Current measuring state
        self.pixmap = None                          # Background image
        self.crosshair_shape = "Crosshair"
        
        self.setMouseTracking(True)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    # New project
    def new_project(self,image_path):
        self.measurement_stack.clear()
        self.measuring_state = None
        self.pixmap = None
        self.scene.clear()

        QtGui.QImageReader.setAllocationLimit(0) # Remove image size limit
        self.pixmap = QtGui.QPixmap(image_path)
        self.scene.addPixmap(self.pixmap)
        self.setSceneRect(QtCore.QRectF(0.0, 0.0, self.pixmap.width(), self.pixmap.height()))   # Set Scenerect to size of pixmap
        self.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.scene.update()

    # Push command(s) to stack
    def push_stack(self, name, measurement_type):
        self.measuring_state = measurement_type
        self.measurement_stack.append(Measurement(measurement_type,name))   # Set object type to item by default
        self.update_application()

    # Pop last command from stack and update graphicsview
    def undo(self):
        if len(self.measurement_stack) > 0:
            last_measurement = self.measurement_stack[-1]         # Grab latest object
            if len(last_measurement.objects_params) == 1 or last_measurement.objects_params[-1]["type"] == consts.PATHITEM or last_measurement.get_type() == consts.WIDTH:
                self.measurement_stack.pop()
                self.measuring_state = None
            elif len(last_measurement.objects_params) > 1:  # If object holds multiple items
                last_measurement.rem_object()           # Remove last item
                self.measuring_state = last_measurement.get_type()    # Update measurement state
                self.update_prev_lineitem(last_measurement)
            else:
                self.measurement_stack.pop()
                self.undo()                # Call self until graphics item is removed or no measurements
                return                      # No need to update scene twice
            self.draw_scene()               # Update scene after update
   
    # Custom clear scene implementation (Does not destroy custom ellipse class from memory)
    def clear_scene(self):
        for item in self.scene.items():
            self.scene.removeItem(item)
   
    # Re-draw QGraphicsView with commands from FIFO stack
    def draw_scene(self):
        self.clear_scene()  # Clear scene
        self.scene.addPixmap(self.pixmap)   # Add pixmap
        
        for measurement in self.measurement_stack:  # For every measurement
            for item in measurement.get_objects():  # For every object in measurement
                match item["type"]:
                    case consts.LINEITEM:
                        self.scene.addItem(QGraphicsLineItem(item["parms"]))
                    case consts.ELLIPSEITEM:
                        item["parms"].update_crosshair(self.slider_pos, self.opacity_pos)
                        self.scene.addItem(item["parms"])
                    case consts.PATHITEM:
                        self.scene.addPath(item["parms"])
                    case consts.POLYGONITEM:
                        ellipse = QGraphicsPolygonItem(item["parms"])
                        ellipse.setBrush(QtGui.QBrush(QtGui.QColor(255,255,255,127)))
                        self.scene.addItem(ellipse)
                    case consts.FONTITEM:
                        font = QFont()
                        font.setPointSize(40)
                        font.setWeight(QFont.Weight.Bold)
                        font.setPixelSize(int(self.pixmap.width()/30))  # Set text size relative to image dimensions
                        textItem = QGraphicsTextItem(item["parms"])
                        textItem.setFont(font)
                        textItem.setPos(item["pos"])
                        self.scene.addItem(textItem)         
        self.update_application()

    # Updates GUI elements (Mouse pointer, toolbar toggles, etc.) after drawing to screen
    def update_application(self):
        # Update mouse cursor
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            QApplication.setOverrideCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        elif self.measuring_state:
            QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)  #change cursor
        else:
            QApplication.setOverrideCursor(QtCore.Qt.CursorShape.ArrowCursor)
        
        # Update Toolbar
        if len(self.measurement_stack) > 0:
            self.parent().enable_undo()
            self.parent().enable_export()
        else:
            self.parent().statusbar.showMessage('Select a measurement to make from the toolbar')
            self.parent().disable_undo()
            self.parent().disable_export()

        self.parent().clear_button_highlights()
        self.parent().disable_all_measurements()
        if self.measuring_state:
            # Disable all buttons but undo
            self.parent().highlight_measurement(self.measuring_state)
        else:  
            # Enable all buttons but width
            self.parent().enable_all_measurements()
            # If last measurement was length
            if len(self.measurement_stack) > 0 and self.measurement_stack[-1].objects_params[-1]["type"] == consts.PATHITEM:
                self.parent().enable_width_measurement()    # Enable width button     

    # Changes size of ellipses drawn on screen
    # Called from MainWindow
    def slider_moved(self, width_value, opactity_value):
        self.slider_pos = width_value
        self.opacity_pos = opactity_value
        self.draw_scene()
                
    # Activated every key press
    def keyPressEvent(self, event):  #shift modifier for panning
        if event.key() == QtCore.Qt.Key.Key_Shift:
            pos = QtGui.QCursor.pos()
            self.dragPos = self.mapToGlobal(self.mapFromGlobal(pos))
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.update_application()
            
    # Return cursor to normal after grabbing screen
    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Shift:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.update_application()

    # PySide event called every mouse move
    def mouseMoveEvent(self, event):
        mousePos = self.mapToScene(event.position().toPoint())
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:    # If User is dragging scene
            delta = mousePos - self.dragPos
            self.translate(delta.x(), delta.y())
        elif self.measuring_state and len(self.measurement_stack) > 0:  # If User is creating a measurement
            cur_measurment = self.measurement_stack[-1]
            match self.measuring_state:
                case consts.AREA:
                    if cur_measurment.has_objects():    # If measurement exists, have p2 follow mouse
                        self.intersect_handler(cur_measurment)      # Check for area intersect
                        self.draw_scene()   # Update scene
                case _:     # Default
                    if cur_measurment.has_objects():    # If measurement exists, have p2 follow mouse
                        self.update_prev_lineitem(cur_measurment)
                        self.draw_scene()   # Update scene

        super().mouseMoveEvent(event)

    # Used by mousemoveevent
    # Checks if QLineF's in area measurement intersect
    # Handles add/removal of area polygon
    def intersect_handler(self, measurement):
        # Check if area ellipse already exists (Should always be last appended item)
        if measurement.objects_params[-1]["type"] == consts.POLYGONITEM:
            measurement.rem_object()

        self.update_prev_lineitem(measurement)  # Update previous line measurement
        
        if len(measurement.objects_params) > 2: # Check if more than 2 lines exist
            last_line = measurement.objects_params[-1]["parms"] # Grab last QLineF
            for i in range(len(measurement.objects_params) -2): # Check if intersects with any line but its neighbor
                inter_check = last_line.intersects(measurement.objects_params[i]["parms"])
                if inter_check[0] == QLineF.IntersectionType.BoundedIntersection:    # Add ellipse
                    points = []
                    for point in measurement.objects_params[i:]:
                        points.append(point["parms"].p2())
                    points.append(inter_check[1])   # Add intersection point
                    measurement.append_object({
                        "parms": QtGui.QPolygonF(points),
                        "type": consts.POLYGONITEM
                    })
                    return  # end intersect check

    # Used by Mousemoveevent
    # Updates point #2 of last lineItem object to mousePos
    def update_prev_lineitem(self, cur_measurment):
        mousePos = self.mapToScene(self.mapFromGlobal(QtGui.QCursor.pos()))
        last_pos = cur_measurment.objects_params[-1]["parms"].p1()
        cur_measurment.objects_params[-1]["parms"] = QtCore.QLineF(last_pos,mousePos)   # Update LineF

    # PySide event called every double click
    def mouseDoubleClickEvent(self, event):
        mousePos = self.mapToScene(event.position().toPoint())
        cur_measurment = self.measurement_stack[-1]
        if self.measuring_state == consts.LENGTH:
            # Check if bezier option is checked
            self.parent().statusbar.showMessage('Length measurement complete.')
            if self.parent().bezier.isChecked():
                last_pos = cur_measurment.objects_params[-1]["parms"].p2()
                cur_measurment.append_object({
                    "parms": QLineF(last_pos,mousePos),
                    "type": consts.LINEITEM
                    })
                self.calculate_curve()
            else:
                self.calculate_length(cur_measurment)
            self.measuring_state = None # Reset to default values
        self.draw_scene()

    # Called every mouse click in scene
    def mousePressEvent(self, event):
        if len(self.measurement_stack) > 0 and self.measuring_state:
            mousePos = self.mapToScene(event.position().toPoint())
            cur_measurment = self.measurement_stack[-1]
            match self.measuring_state:
                case consts.LENGTH:
                    self.add_line_item(cur_measurment,mousePos)
                case consts.ANGLE:
                    if cur_measurment.has_objects() and len(cur_measurment.get_objects()) >= 2:
                        self.calculate_angle(cur_measurment)
                        self.parent().statusbar.showMessage('Angle measurement complete')
                        self.measuring_state = None
                    else:
                        self.add_line_item(cur_measurment,mousePos)
                case consts.AREA:
                    if cur_measurment.has_objects() and cur_measurment.objects_params[-1]["type"] == consts.POLYGONITEM:
                        self.calculate_area(cur_measurment)
                        self.parent().statusbar.showMessage('Polygon area measurement completed')
                        self.measuring_state = None
                    else:
                        self.add_line_item(cur_measurment,mousePos)
            self.draw_scene()
        super().mousePressEvent(event)

    # Adds lineItem to measurement
    def add_line_item(self,measurement,mousePos):
        if measurement.has_objects():
            last_pos = measurement.objects_params[-1]["parms"].p2()
            measurement.append_object({
                "parms": QLineF(last_pos,mousePos),
                "type": consts.LINEITEM
                })
        else:
            measurement.append_object({
                "parms": QLineF(mousePos,mousePos),
                "type": consts.LINEITEM
                })

    # Used to calculate and draw bezier curve
    def calculate_curve(self):
        measurement = self.measurement_stack[-1]

        L = posData(
                np.empty(shape=(0, 0)),
                np.empty(shape=(0, 0)))
        top_object = measurement.objects_params[0]["parms"]

        L.update(top_object.p1().x(),top_object.p1().y())
        for object in measurement.get_objects():    # Grab all remaining points
            L.update(object["parms"].p2().x(),object["parms"].p2().y())

        nt = 100 #max(1000, self.numwidths * 50)  #num of interpolating points
        t = np.linspace(0.0, 1.0, nt)
        P = np.vstack((L.x, L.y)).T #control points
        kb = len(P) - 1 #order of bezier curve # of control points (n) - 1

        B = bezier(t, P, k = kb) #evaluate bezier curve along t
        Q = kb*np.diff(P, axis = 0)
        
        measurement.measurement_value = gauss_legendre(b = 1, f = bezier, P = Q, k = kb - 1, arc = True) #compute total arc length.
        measurement.Q = Q
        measurement.kb = kb
        measurement.P = P
        measurement.objects_params.clear()                    # Clear object array for curve

        xs, ys = B[:,0], B[:,1]

        for i in range(1, nt - 1):
            P0 = QtCore.QPointF( xs[i-1], ys[i-1] )#.toPoint()
            P1 = QtCore.QPointF( xs[i  ], ys[i  ] )#.toPoint()
            P2 = QtCore.QPointF( xs[i+1], ys[i+1] )#.toPoint()

            path = QtGui.QPainterPath(P0)
            path.cubicTo(P0, P1, P2)
            measurement.append_object({
                        "parms": path,
                        "type": consts.PATHITEM
                        })
            
    # Calculate total length of length measurement
    def calculate_length(self, measurement):
        measurement.measurement_value = sum([l["parms"].length() for l in measurement.objects_params])

    # Calculate qreal (PySide degrees) of a finished angle measurement
    def calculate_angle(self, measurement):
        lines = measurement.get_objects()
        measurement.measurement_value = lines[0]["parms"].angleTo(lines[1]["parms"])

    def calculate_area(self, measurement):
        qpolygon = measurement.objects_params[-1]["parms"]

        # Shoelace formula: https://www.theoremoftheday.org/GeometryAndTrigonometry/Shoelace/TotDShoelace.pdf
        S1 = sum((qpolygon[i].x()*qpolygon[i+1].y())-(qpolygon[i].y()*qpolygon[i+1].x()) for i in range(len(qpolygon)-1))
        conct = (qpolygon[-1].x()*qpolygon[0].y())-(qpolygon[-1].y()*qpolygon[0].x())
        measurement.measurement_value = 0.5*abs(S1+conct)

    # Calculates distance in pixels of wdiths measurement
    # Calculate on export due to MovingEllipse being a dynamic item
    def calculate_widths(self,bias):
        for measurement in self.measurement_stack:  # For every measurement
            if measurement.get_type() == consts.WIDTH:  # Find width measurements
                width_array = []
                side_A_width = []
                side_B_width = []
                for item in measurement.get_objects():
                    if item["type"] == consts.ELLIPSEITEM:
                        match item["parms"].side:
                            case consts.SIDE_A:
                                side_A_width.append(item["parms"])
                            case consts.SIDE_B:
                                side_B_width.append(item["parms"])
    
                for A,B in zip(side_A_width,side_B_width):
                    match bias:
                        case "Side A":
                            width_array.append(QLineF(A.scenePos(), A.centerLinePoint).length())
                        case "Side B":
                            width_array.append(QLineF(B.scenePos(),B.centerLinePoint).length())
                        case _: # None
                            width_array.append(QLineF(A.scenePos(),B.scenePos()).length())
                measurement.measurement_value = width_array    # Calculated in pixels

    # Iterates over measurement stack to return names and values of measurements
    def get_measurement_names_and_values(self, m):
        pixel_measurement = []
        unit_measurement = []   # meters
        for measurement in self.measurement_stack:
            match measurement.get_type():
                case consts.WIDTH:
                    num_widths = len(measurement.measurement_value)
                    for i in range(num_widths):
                        pixel_measurement.append([measurement.get_name()+"_w"+"{0:.1f}".format((i+1)/(num_widths+1)*100),"{0:.2f}".format(measurement.measurement_value[i]), "Pixels"])
                        unit_measurement.append([measurement.get_name()+"_w"+"{0:.1f}".format((i+1)/(num_widths+1)*100), "{0:.2f}".format(measurement.measurement_value[i]*m), "Meters"])
                case consts.LENGTH:
                    pixel_measurement.append([measurement.get_name(),"{0:.2f}".format(measurement.measurement_value), "Pixels"])
                    unit_measurement.append([measurement.get_name(), "{0:.2f}".format(measurement.measurement_value*m), "Meters"])
                case consts.ANGLE:
                    pixel_measurement.append([measurement.get_name(), "{0:.2f}".format(measurement.measurement_value), "Degrees"])
                    unit_measurement.append([measurement.get_name(), "{0:.2f}".format(measurement.measurement_value), "Degrees"])
                case consts.AREA:
                    pixel_measurement.append([measurement.get_name(), "{0:.2f}".format(measurement.measurement_value), "Pixels"])
                    unit_measurement.append([measurement.get_name(), "{0:.2f}".format(measurement.measurement_value*(m**2)), "Square Meters"])
        return pixel_measurement, unit_measurement

    # Measure widths of aquatic animal (Called when GUI button is pressed)
    def measure_widths(self):        
        if len(self.measurement_stack) > 0 and self.measurement_stack[-1].measurement_type == consts.LENGTH:
            self.parent().statusbar.showMessage('Drag width segment points to make width measurements perpendicular to the length segment')
            last_measurement = self.measurement_stack[-1]
            self.push_stack(last_measurement.get_name(), 4)
            width_measurement = self.measurement_stack[-1]
            numwidths = int(self.parent().subWin.numwidths.text())-1
            k = 0

            s_i = np.linspace(0,1,numwidths+2)[1:-1]    #only need to draw widths for inner pts
            t_i = np.array([root_scalar(gauss_legendre, x0 = s_i, bracket = [-1,1], method = "bisect", args = (bezier, last_measurement.Q, last_measurement.kb-1, True, s, last_measurement.measurement_value) ).root for s in s_i])
            B_i = bezier(np.array(t_i), P = last_measurement.P, k = last_measurement.kb)

            #Find normal vectors by applying pi/2 rotation matrix to tangent vector
            bdot = bezier(t_i, P = last_measurement.Q, k = last_measurement.kb - 1)
            mag = np.linalg.norm(bdot,axis = 1) #normal vector magnitude
            bnorm = np.flip(bdot/mag[:,None],axis = 1)
            bnorm[:,0] *= -1
            
            for k,(pt,m) in enumerate(zip(B_i,bnorm)):
                x1, y1 = pt[0],pt[1]
                vx, vy = m[0], m[1]
                L = self.pixmap.width()
                H = self.pixmap.height()

                xi,yi = [], []
                for bound in ([0,0],[L,H]):
                    for ev in ([1,0],[0,1]):
                        A = np.matrix([ [vx, ev[0]] , [vy, ev[1]] ])
                        b = np.array([bound[0] - x1, bound[1] - y1])
                        T = np.linalg.solve(A,b)[0] #only need parametric value for our vector, not bound vector

                        xint = x1 + T*vx
                        yint = y1 + T*vy
                        if ( (xint<=L + 1) and (xint>=0-1) and (yint<=H+1) and (yint>=0-1) ): #only add intersect if conditions met.
                            #1 pixel fudge factor required?
                            xi.append(xint)
                            yi.append(yint)

                # Draw width lines (And draw starting points)
                for l, (x, y) in enumerate(zip(xi,yi)):

                    start = QtCore.QPointF(x1, y1)
                    end = QtCore.QPointF(x, y)

                    # if this is the first itertion
                    if k == 0:
                        # Set distance from linear measurement
                        lineLength = np.sqrt((start.x()-end.x())**2 + (start.y()-end.y())**2)
                        t = (500)/lineLength # Ratio of desired distance from center / total length of line
                        posAB = QtCore.QPointF(((1-t)*start.x()+t*end.x()),((1-t)*start.y()+t*end.y()))
                        if l == 0:
                            width_measurement.append_object({
                                "parms": str("A"),
                                "pos": posAB,
                                "type": consts.FONTITEM
                            })
                        elif l == 1:
                            width_measurement.append_object({
                                "parms": str("B"),
                                "pos": posAB,
                                "type": consts.FONTITEM
                            })
                    width_measurement.append_object({   # Append Ellipse
                                "parms": MovingEllipse(self, start, end, self.slider_pos, l%2, self.parent().subWin.width_tabs.currentText()),
                                "type": consts.ELLIPSEITEM
                            })
                    width_measurement.append_object({   # Append Line
                                "parms": QtCore.QLineF(start, end),
                                "type": consts.LINEITEM
                            })
        self.measuring_state = None
        self.draw_scene()

    # Hot key event
    def polyClose(self):
        pass

    # call super proxy for child classes (Ellipse)
    def hoverEnterEvent(self, event):
        super().hoverEnterEvent(event)

    # call super proxy for child classes (Ellipse)
    def hoverLeaveEvent(self, event):
        super().hoverLeaveEvent(event)

    # call super proxy for child classes (Ellipse)
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

    # Zoom
    def wheelEvent(self, event):
        #https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview
        #transform coordinates correctly
        #https://stackoverflow.com/questions/20942586/controlling-the-pan-to-anchor-a-point-when-zooming-into-an-image
        #https://stackoverflow.com/questions/41226194/pyqt4-pixel-information-on-rotated-image
        zoomInFactor = 1.05
        zoomOutFactor = 1 / zoomInFactor

        # self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        # self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        oldPos = self.mapToScene(event.position().toPoint())

        #Zoom
        # https://quick-geek.github.io/answers/885796/index.html
        # y-component for mouse with two wheels
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)

        newPos = self.mapToScene(event.position().toPoint())  #Get the new position
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())  #Move scene to old position
        

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(__file__)

    return os.path.join(base_path, relative_path)

# Ellipse QGraphicsItem Class
# A grabable object to change width measurements dynamically
# Ellipse is bound to parent line
# Input: Point P1 (QPointF), Point P2 (QPointF)
class MovingEllipse(QGraphicsPixmapItem):
    def __init__(self, parent,lp1, lp2, scale, side, shape_type):
        # LP2 is always border point (PyQt6.QtCore.QPointF(1030.9353133069922, 0.0))
        super(MovingEllipse,self).__init__()

        scaledSize = 10 + (scale*10)
        self.shape_key = shape_type
        match self.shape_key:
            case "Crosshair":   # Save user selected shape type
                Image = QPixmap(resource_path("crosshair.png")).scaled(scaledSize,scaledSize)
            case "Dot":
                Image = QPixmap(resource_path("dot.png")).scaled(scaledSize,scaledSize)
        
        self.color = parent.picked_color
        self.setOpacity(parent.opacity_pos)
        self.Pixmap = QPixmap(Image.size())
        self.Pixmap.fill(self.color)
        self.Pixmap.setMask(Image.createMaskFromColor(Qt.GlobalColor.transparent))

        self.setPixmap(self.Pixmap)
        self.setOffset(QtCore.QPointF(-scaledSize/2,-scaledSize/2)) # Set offset to center of image
        
        self.centerLinePoint = lp1  # Used in width measurement
        self.side = side
        self.p1 = lp1   # Boundry points
        self.p2 = lp2

        self.parent = parent            # Used for updating widths measurement
        # Find slope of line (y2-y1)/(x2-x1)
        self.m = (self.p2.y()-self.p1.y())/(self.p2.x()-self.p1.x())
        self.assignPoints(self.m,lp1,lp2)
        # Find X intercept
        self.y0 = (lp1.y())-(self.m*lp1.x())  # y1-(m*x1) = b
        self.x0 = (self.y0*-1)/self.m   # -y0/slope

        # Set distance from linear measurement
        d = np.sqrt((lp1.x()-lp2.x())**2 + (lp1.y()-lp2.y())**2)
        t = (scaledSize*3)/d # Ratio of desired distance from center / total length of line

        self.setPos(QtCore.QPointF(((1-t)*lp1.x()+t*lp2.x()),((1-t)*lp1.y()+t*lp2.y())))
        self.setAcceptHoverEvents(True)
        self.drag = False

    def update_crosshair(self, scale, opacity):
        # scaledSize = int(self.parent.scene.height()/60) + (scale*10) # OLD WAY, just set to 50 pixel minimum for those hardcore low res users
        scaledSize = 10 + (scale*10)
        match self.shape_key:
            case "Crosshair":   # Save user selected shape type
                Image = QPixmap(resource_path("crosshair.png")).scaled(scaledSize,scaledSize)
            case "Dot":
                Image = QPixmap(resource_path("dot.png")).scaled(scaledSize,scaledSize)
        self.Pixmap = QPixmap(Image.size())
        self.Pixmap.fill(self.color)
        self.Pixmap.setMask(Image.createMaskFromColor(Qt.GlobalColor.transparent))

        self.setPixmap(self.Pixmap)
        self.setOffset(QtCore.QPointF(-scaledSize/2,-scaledSize/2)) # Set offset to center of image
        self.setOpacity(opacity/10)

    def assignPoints(self, slope, lp1, lp2):
        # Set Points depending on their path slope 
        if slope > -0.5 and slope < 0.5:    # Point assignmnet dependent on starting X values if slope is large
            if lp1.x() < lp2.x():
                self.p1 = lp1               # P1 should always hold the smaller comparetor
                self.p2 = lp2
            else:
                self.p1 = lp2
                self.p2 = lp1
        else:                           
            if lp1.y() < lp2.y():
                self.p1 = lp1
                self.p2 = lp2
            else:
                self.p1 = lp2
                self.p2 = lp1

    # Mouse Hover
    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    # Mouse Stops Hovering
    def hoverLeaveEvent(self, event):
        QApplication.setOverrideCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event):
        self.drag = True
        QApplication.setOverrideCursor(QtCore.Qt.CursorShape.BlankCursor)

    def mouseMoveEvent(self, event):
        if self.drag:
            orig_curs_pos = event.lastScenePos()
            updated_curs_pos = event.scenePos()
            orig_pos = self.scenePos()

            # Update position of Ellipse to match mouse

            ell_y = updated_curs_pos.y()# - orig_curs_pos.y() + orig_pos.y()
            ell_x = updated_curs_pos.x()# - orig_curs_pos.x() + orig_pos.x()

            # Use X of mouse when line is horizontal, and Y when verticle
            if self.m > -0.5 and self.m < 0.5:
                # y = mx + b
                if ell_x < self.p1.x():
                    ell_x = self.p1.x()
                elif ell_x > self.p2.x():
                    ell_x = self.p2.x()
                ell_y = ell_x*self.m + self.y0
                
            else:
                # x = y/m - r
                # Stay within boundry
                if ell_y < self.p1.y():
                    ell_y = self.p1.y()
                elif ell_y > self.p2.y():
                    ell_y = self.p2.y()

                ell_x = ell_y/self.m + self.x0
                
            self.setPos(QtCore.QPointF(ell_x, ell_y))

    def mouseReleaseEvent(self, event):
        self.drag = False
        QApplication.setOverrideCursor(QtCore.Qt.CursorShape.ArrowCursor)

# Calculate bezier function for line fitment
def bezier(t,P,k,arc = False):
    """
    Matrix representation of Bezier curve following
    https://pomax.github.io/bezierinfo/#arclength
    """
    signs = np.array([i for j,i in zip(range(k+1),islice(cycle([1, -1]),0,None))]) #create array alternating 0,1s for diagonals
    A = pascal(k+1, kind='lower') #generate Pascal triangle matrix
    S = diags(signs, [i-k for i in range(k+1)][::-1], shape=(k+1, k+1)).toarray() #create signs matrix
    M = A*S #multiply pascals by signs to get Bernoulli polynomial matrix
    coeff = A[-1,:]
    C = M*coeff[:,None] #broadcast
    T = np.array( [t**i for i in range(k+1)] ).T

    B = T.dot( C.dot(P) )

    if arc:
        return np.linalg.norm(B, axis = 1)
    else:
        return B

# Gauss-Legendre Quadrature for bezier curve arc length
def gauss_legendre(b, f, P, k, arc, loc = 0.0, L = 1, degree = 24, a = 0):
    x, w = np.polynomial.legendre.leggauss(degree)
    t = 0.5*(b-a)*x + 0.5*(b+a)

    return 0.5*(b-a)*np.sum( w*bezier(t,P,k,arc) )/L - loc

# Class to hold measurement of objects and current states for prcedural QGraphicScene
class Measurement():

    def __init__(self, mt, name):
        self.measurement_type = mt
        self.measurement_name = name
        self.objects_params = []
        self.measurement_value = None

        # Items used by width measurement
        self.Q = None
        self.kb = None
        self.l = None
        self.P = None

    # Return dict containing state of scene at time of measurement
    def get_type(self):
        return self.measurement_type
    
    # Return list of Qt objects stored
    def get_objects(self):
        return self.objects_params
    
    # Return name of measurement
    def get_name(self):
        return self.measurement_name
    
    # Append Qt object to measurement class
    def append_object(self, object):
        self.objects_params.append(object)  # Create deepcopy

    # Remove top object
    def rem_object(self):
        self.objects_params.pop()

    # Return True/False if class holds objects
    def has_objects(self):
        if len(self.objects_params) > 0:
            return True
        return False
        
# Custom class to hold position data of QPoint
class posData():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def update(self, add_x, add_y):
        self.x = np.append(self.x, add_x)
        self.y = np.append(self.y, add_y)

        #below just for area calcs
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.Tu = np.hypot(self.dx,self.dy) + np.finfo(float).eps