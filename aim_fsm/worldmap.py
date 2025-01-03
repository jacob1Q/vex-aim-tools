from .geometry import *

class WorldObject():
    def __init__(self, id=None, x=0, y=0, z=0, is_visible=False):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.is_fixed = False   # True for walls and markers in predefined maps
        self.is_obstacle = True
        self.is_visible = is_visible
        self.is_foreign = False
        if is_visible:
            self.pose_confidence = +1
        else:
            self.pose_confidence = -1

    def __repr__(self):
        vis = "visible" if self.is_visible else "unseen"
        return f'<{self.__class__.__name__} {vis} at ({self.x:.1f}, {self.y:.1f})>'


class OrangeBarrelObj(WorldObject):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.name = spec['name']
        self.diameter = 22 # mm

class BlueBarrelObj(WorldObject):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.name = spec['name']
        self.diameter = 22 # mm

class BallObj(WorldObject):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.name = spec['name']
        self.diameter = 25.0 # mm
        self.z = self.diameter / 2

class RobotObj(WorldObject):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.name = spec['name']

class AprilTagObj(WorldObject):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.name = spec['name']
        self.tag_id = spec['id']
        self.theta = spec['angle'] / 180 * pi
        self.diameter = 22 # mm

    def __repr__(self):
        vis = "visible" if self.is_visible else "unseen"
        return f'<{self.__class__.__name__} id={self.tag_id} {vis} at ({self.x:.1f}, {self.y:.1f}) @ {self.theta*180/pi:.1f} deg.>'

class ArucoMarkerObj(WorldObject):
    def __init__(self, spec, x=0, y=0, z=0, theta=0):
        super().__init__(self)
        self.name = spec['name']
        self.tag_id = spec['id']
        self.aruco_parent = spec['marker'].aruco_parent
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.pose_confidence = +1

    def __repr__(self):
        if self.pose_confidence >= 0:
            vis = ' visible' if self.is_visible else ''
            fix = ' fixed' if self.is_fixed else ''
            return '<ArucoMarkerObj %d: (%.1f, %.1f, %.1f) @ %d deg.%s%s>' % \
                (self.tag_id, self.x, self.y, self.z, self.theta*180/pi, fix, vis)
        else:
            return '<ArucoMarkerObj %d: position unknown>' % self.tag_id

################################################################

class WorldMap():

    def __init__(self,robot):
        self.robot = robot
        self.objects = dict()
        self.shared_objects = dict()

    def __repr__(self):
        return f'<WorldMap with {len(self.objects)} objects>'

    def clear(self):
        self.objects.clear()
        #self.robot.world.particle_filter.clear_landmarks()

    def update(self):
        self.seen_objs = []
        self.update_aivision_objects()
        if self.robot.aruco_detector:
            self.update_aruco_objects()
        for obj in self.objects.values():
            if obj not in self.seen_objs:
                obj.is_visible = False

    def update_aivision_objects(self):
        objspecs = self.robot.robot0._ws_status_thread.current_status['aivision']['objects']['items']
        for spec in objspecs:
            if spec['type_str'] == 'aiobj':
                name = spec['name']
            elif spec['type_str'] == 'tag':
                if 0 <= spec['id'] <= 4:
                    name = 'AprilTag-' + repr(spec['id'])
                    spec['name'] = name
                else:
                    #print('*** BAD TAG:', spec)
                    continue
            else:
                print(f'spec={spec}')
                continue
            if name not in self.objects:
                obj = self.make_object(spec)
                self.objects[obj.name] = obj
                print(f"Created {obj}")
            else:
                obj = self.objects[name]
            obj.is_visible = True
            self.seen_objs.append(obj)
            self.update_aivision_object_position(spec)

    def update_aruco_objects(self):
        for (id,marker) in self.robot.aruco_detector.seen_marker_objects.items():
            name = f'ArucoMarker-{id}'
            spec = {'name': name, 'id': id, 'marker': marker}
            if name not in self.objects:
                obj = self.make_object(spec)
                self.objects[obj.name] = obj
            else:
                obj = self.objects[name]
            obj.is_visible = True
            self.seen_objs.append(obj)
            self.update_aruco_object_position(spec)

    def make_object(self, spec):
        if spec['name'] == 'OrangeBarrel':
            obj = OrangeBarrelObj(spec)
        elif spec['name'] == 'BlueBarrel':
            obj = BlueBarrelObj(spec)
        elif spec['name'] == 'Ball':
            obj = BallObj(spec)
        elif spec['name'] == 'Robot':
            obj = RobotObj(spec)
        elif spec['name'].startswith('AprilTag'):
            obj = AprilTagObj(spec)
        elif spec['name'].startswith('ArucoMarker'):
            obj = ArucoMarkerObj(spec)
        else:
            print(f"ERROR **** spec = {spec}")
            obj = None
        return obj

    def update_aivision_object_position(self, spec):
        resolution_scale = 2
        cx = (spec['originx'] + spec['width']/2) * resolution_scale
        cy = (spec['originy'] + spec['height']) * resolution_scale
        obj = self.objects[spec['name']]
        if isinstance(obj, AprilTagObj):
            cy += spec['height'] * 2 * resolution_scale
        hit = self.robot.kine.project_to_ground(cx, cy)
        # offset hit by half the object thickness
        if obj.__dict__.get('diameter'):
            hit += point(obj.diameter / 2, 0, 0)
        # convert to world coordinates
        robotpos = point(self.robot.x, self.robot.y)
        objpos = aboutZ(self.robot.theta).dot(hit) + robotpos
        obj.x = objpos[0][0]
        obj.y = objpos[1][0]
        tag_angle_correction_factor = 4  # guesstimate
        if spec.get('angle') != None:
            angle = spec['angle'] - (0 if spec['angle'] < 180 else 360)
            obj.theta = self.robot.theta - angle / 180 * pi * tag_angle_correction_factor

    def update_aruco_object_position(self, spec):
        obj = self.objects[spec['name']]
        marker = spec['marker']
        sensor_dist = marker.camera_distance
        sensor_coords = marker.camera_coords
        sensor_bearing = atan2(sensor_coords[0], sensor_coords[2])
        sensor_orient = wrap_angle(pi - marker.euler_rotation[1] * (pi/180))
        theta = self.robot.theta
        obj.x = self.robot.x + sensor_dist * cos(theta + sensor_bearing)
        obj.y = self.robot.y + sensor_dist * sin(theta + sensor_bearing)
        obj.z = marker.aruco_parent.marker_size / 2  # *** TEMPORARY HACK ***
        obj.theta = wrap_angle(self.robot.theta - sensor_orient)

