"""
OpenGL-Based Camera Viewer
"""

import numpy as np

try:
    import cv2
    from OpenGL.GLUT import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
except:
    pass

from . import opengl
from .camera import AIVISION_RESOLUTION_SCALE

# For capturing images
global snapno, path, running_fsm
snapno = 0
path = 'snapshots/'

WINDOW = None

class CamViewer():
    def __init__(self, robot, width=640, height=480, user_annotate_function=None,
                 windowName="Robot View"):
        self.robot = robot
        self.width = width
        self.height = height
        self.aspect = self.width/self.height
        self.user_annotate_function = user_annotate_function
        self.windowName = windowName
        self.scale = 1
        self.show_axes = True
        self.crosshairs = False

    def process_image(self):
        raw = self.robot.camera_image

        if self.scale == 1:
            image = raw.copy()
        else:
            shape = raw.shape
            dsize = (self.scale*shape[1], self.scale*shape[0])
            image = cv2.resize(raw, dsize)

        if self.crosshairs:
            cv2.line(image, (int(self.width/2), 0), (int(self.width/2), self.height), (255,255,0), 1)
            cv2.line(image, (0, int(self.height/2)), (self.width, int(self.height/2)), (255,255,0), 1)

        for obj in self.robot.robot0.status['aivision']['objects']['items']:
            name = obj.get('name', None)
            if name == 'SportsBall':
                color = (255, 255, 0)
            elif name == 'OrangeBarrel':
                color = (255, 50, 50)
            elif name == 'BlueBarrel':
                color = (50, 100, 255)
            elif name == 'Robot':
                color = (255, 255, 255)
            else:
                color = (0, 255, 0)
            if obj['type_str'] == 'aiobj':
                cv2.rectangle(image,
                              (obj['originx']*AIVISION_RESOLUTION_SCALE,
                               obj['originy']*AIVISION_RESOLUTION_SCALE),
                              ((obj['originx'] + obj['width'])*AIVISION_RESOLUTION_SCALE,
                               (obj['originy'] + obj['height'])*AIVISION_RESOLUTION_SCALE),
                              color,
                              1)
            elif obj['type_str'] == 'tag':
                corners = np.array(((obj['x0'], obj['y0']),
                                    (obj['x1'], obj['y1']),
                                    (obj['x2'], obj['y2']),
                                    (obj['x3'], obj['y3'])),
                                   np.int32) * AIVISION_RESOLUTION_SCALE
                corners = corners.reshape(-1,1,2)
                cv2.polylines(image,
                              [corners],
                              True,  # closed curve
                              color)
            else:
                print('*** CamViewer:', obj)
        if self.robot.aruco_detector and len(self.robot.aruco_detector.seen_marker_ids) > 0:
            self.robot.aruco_detector.annotate(image, self.scale)
        if self.user_annotate_function:
            image = self.user_annotate_function(image)
        self.robot.annotated_image = image
        # Done with annotation
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
        glutPostRedisplay()

    # ================ Window Setup ================
    def window_creator(self):
        global WINDOW
        #glutInit(sys.argv)
        WINDOW = opengl.create_window(
            bytes(self.windowName, 'utf-8'), (self.width, self.height))
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(100, 100)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyPressed)
        glutSpecialFunc(self.specialKeyPressed)
        glutSpecialUpFunc(self.specialKeyUp)

    def start(self):  # Displays in background
        if not WINDOW:
            opengl.init()
            opengl.CREATION_QUEUE.append(self.window_creator)

    def display(self):
        self.process_image()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        # Set Projection Matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)

        glMatrixMode(GL_TEXTURE)
        glLoadIdentity()
        glScalef(1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(0.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(self.width, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(self.width, self.height)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(0.0, self.height)
        glEnd()

        glFlush()
        glutSwapBuffers()

    def reshape(self, w, h):
        if h == 0:
            h = 1

        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)

        glLoadIdentity()
        nRange = 1.0
        if w <= h:
            glOrtho(-nRange, nRange, -nRange*h/w, nRange*h/w, -nRange, nRange)
        else:
            glOrtho(-nRange*w/h, nRange*w/h, -nRange, nRange, -nRange, nRange)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def keyPressed(self, key, x, y):
        if ord(key) == 27:
            print("Use 'exit' to quit.")
            return
        elif key == b's':
            print("Taking a raw snap")
            self.capture_raw()
        elif key == b'S':
            print("Taking an annotated snap")
            self.capture_annotated()
        elif key == b'c':
            self.crosshairs = not self.crosshairs
        elif key == b'h':
            print(self.keyboard_help)
        self.display()

    def specialKeyPressed(self, key, x, y):
        global leftorrightindicate, globthres
        if key == GLUT_KEY_LEFT:
            #self.robot.drive_wheels(-100, 100)
            leftorrightindicate = True
            globthres=100
        elif key == GLUT_KEY_RIGHT:
            #self.robot.drive_wheels(100, -100)
            leftorrightindicate = True
            globthres = 100
        elif key == GLUT_KEY_UP:
            #self.robot.drive_wheels(200, 200)
            leftorrightindicate = False
            globthres = 100
        elif key == GLUT_KEY_DOWN:
            #self.robot.drive_wheels(-200, -200)
            leftorrightindicate = True
            globthres = 100
        glutPostRedisplay()

    def specialKeyUp(self, key, x, y):
        global leftorrightindicate, go_forward
        #self.robot.drive_wheels(0, 0)
        leftorrightindicate = True
        go_forward = GLUT_KEY_UP
        glutPostRedisplay()

    def capture_raw(self, name='robot_snap'):
        self.capture_image(self.robot.camera_image, name)

    def capture_annotated(self, name='robot_asnap'):
        self.capture_image(self.robot.annotated_image, name)

    def capture_image(self, image, name):
        global snapno, path
        if not os.path.exists(path):
                os.makedirs(path)
        filename = f"{path}{name}{snapno}.png"
        swapped_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        res = cv2.imwrite(filename, swapped_image)
        print(f"Wrote {filename} with result {res}")
        snapno +=1

    keyboard_help = """
Camera viewer help:
    Type 'c' to toggle crosshairs.
    Type 's' to take a snapshot of the raw camera image.
    Type 'S' to take an annotated snapshot (includes bounding boxes and crosshairs).

"""
