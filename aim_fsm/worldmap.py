import numpy as np

from .geometry import *

# aivision currently uses 320x240 dimensions
AIVISION_RESOLUTION_SCALE = 2

class WorldObject():
    def __init__(self, id=None, x=0, y=0, z=0, is_visible=False):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.name = self.__class__.__name__
        self.matched = None  # matching objevt from data association
        # insert EKF data here
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
        return f'<{self.name} {vis} at ({self.x:.1f}, {self.y:.1f})>'

    def update_matched_object(self):
        # Should use our x/y/z to update EKF values of object in self.matched.
        # For now just do a stupid averaging operation.
        self.matched.x = (self.matched.x + self.x) / 2
        self.matched.y = (self.matched.y + self.y) / 2
        self.matched.z = (self.matched.z + self.z) / 2
        self.matched.is_visible = True

class BarrelObj(WorldObject):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.name = spec['name']
        self.diameter = 22 # mm

class OrangeBarrelObj(BarrelObj):
    pass

class BlueBarrelObj(BarrelObj):
    pass

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
        self.name_counts = dict()

    def __repr__(self):
        return f'<WorldMap with {len(self.objects)} objects>'

    def clear(self):
        self.objects.clear()

    def update(self):
        self.updated = []
        self.make_new_objects_from_vision()
        self.associate_objects()
        self.update_associated_objects()
        self.add_unassociated_objects()
        self.update_visibilities()
        # Consider deletion of unmatched worldmap objects
        # Update robot's pose if we have landmarks available

    def make_new_objects_from_vision(self):
        self.candidates = list()
        self.make_new_aiv_objects()
        if self.robot.aruco_detector:
            self.make_new_aruco_objects()

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

    def make_new_aiv_objects(self):
        objspecs = self.robot.robot0.status['aivision']['objects']['items']
        for spec in objspecs:
            if spec['type_str'] == 'aiobj':
                base_name = spec['name']
            elif spec['type_str'] == 'tag':
                if 0 <= spec['id'] <= 4:
                    base_name = 'AprilTag-' + repr(spec['id'])
                    spec['name'] = base_name
                else:
                    #print('*** BAD TAG:', spec)
                    continue
            else:
                print(f'*** Unknown: spec={spec}')
                continue
            obj = self.make_object(spec)
            # Calculate midpoint of bottom edge, which we assume is on the floor
            cx = (spec['originx'] + spec['width']/2) * AIVISION_RESOLUTION_SCALE
            cy = (spec['originy'] + spec['height']) * AIVISION_RESOLUTION_SCALE
            if isinstance(obj, AprilTagObj):
                cy += spec['height'] * 2 * AIVISION_RESOLUTION_SCALE
            hit = self.robot.kine.project_to_ground(cx, cy)
            # offset hit by half the object thickness
            if obj.__dict__.get('diameter'):
                hit += point(obj.diameter / 2, 0, 0)
            # convert to world coordinates
            robotpos = point(self.robot.x, self.robot.y)
            objpos = aboutZ(self.robot.theta).dot(hit) + robotpos
            obj.x = objpos[0][0]
            obj.y = objpos[1][0]
            if isinstance(obj, AprilTagObj):
                tag_angle_correction_factor = 4  # guesstimate
                angle = spec['angle'] - (0 if spec['angle'] < 180 else 360)
                obj.theta = self.robot.theta - angle / 180 * pi * tag_angle_correction_factor
            self.candidates.append(obj)

    def make_new_aruco_objects(self):
        for (id,marker) in self.robot.aruco_detector.seen_marker_objects.items():
            name = f'ArucoMarker-{id}'
            spec = {'name': name, 'id': id, 'marker': marker}
            sensor_dist = marker.camera_distance
            sensor_coords = marker.camera_coords
            sensor_bearing = atan2(sensor_coords[0], sensor_coords[2])
            sensor_orient = wrap_angle(pi - marker.euler_rotation[1] * (pi/180))
            theta = self.robot.theta
            obj = self.make_object(spec)
            obj.x = self.robot.x + sensor_dist * cos(theta + sensor_bearing)
            obj.y = self.robot.y + sensor_dist * sin(theta + sensor_bearing)
            obj.z = marker.aruco_parent.marker_size / 2  # *** TEMPORARY HACK ***
            obj.theta = wrap_angle(self.robot.theta - sensor_orient)
            self.candidates.append(obj)

    def associate_objects(self):
        obj_types = list(set(type(obj) for obj in self.candidates))
        for otype in obj_types:
            self.associate_objects_of_type(otype)

    def associate_objects_of_type(self, otype):
        def association_cost(obj1, obj2):
            return ((obj1.x-obj2.x)**2 + (obj1.y-obj2.y)**2)
        new = [c for c in self.candidates if type(c) is otype]
        old = [o for o in self.objects.values() if type(o) is otype]
        N_new = len(new)
        N_old = len(old)
        if N_old == 0:
            return
        costs = np.zeros([N_new,N_old])
        for i in range(N_new):
            for j in range(N_old):
                costs[i,j] = association_cost(new[i], old[j])
        #*** Stupid greedy algorithm; replace with the Hungarian algorithm
        MAX_ACCEPTABLE_COST = 200
        for i in range(N_new):
            bestj = costs[i,:].argmin()
            if costs[i,bestj] < MAX_ACCEPTABLE_COST:
                new[i].matched = old[bestj]
                costs[:,bestj] = 1 + MAX_ACCEPTABLE_COST

    def update_associated_objects(self):
        for candidate in self.candidates:
            if candidate.matched:
                candidate.update_matched_object()
                self.updated.append(candidate.matched)

    def add_unassociated_objects(self):
        for candidate in self.candidates:
            if candidate.matched is None:
                candidate.name = self.next_in_sequence(candidate.name)
                self.objects[candidate.name] = candidate
                candidate.is_visible = True
                self.updated.append(candidate)
                print('Added', candidate)

    def next_in_sequence(self,name):
        count = 1 + self.name_counts.get(name, 0)
        self.name_counts[name] = count
        return name + "." + self.to_base_26(count)

    def to_base_26(self, num):
        result = []
        while num > 0:
            num -= 1  # Adjust for 1-based indexing (A=1, Z=26)
            remainder = num % 26
            result.append(chr(remainder + ord('a')))
            num //= 26
            return ''.join(reversed(result))
    
    def update_visibilities(self):
        for obj in self.objects.values():
            if obj not in self.updated:
                obj.is_visible = False


################ GPT interface ################

    def get_prompt(self):
        def neaten(x):
            return round(x*10)/10
        prompt = ''
        prompt += f'You are located at ({neaten(self.robot.x)}, {neaten(self.robot.y)})\n'
        prompt += f'Your heading is {neaten(self.robot.theta*180/pi)} degrees\n'
        for (key,value) in self.objects.items():
            prompt += f'{key} is located at ({neaten(value.x)}, {neaten(value.y)}) and is {"visible" if value.is_visible else "not visible"}\n'
        return prompt
