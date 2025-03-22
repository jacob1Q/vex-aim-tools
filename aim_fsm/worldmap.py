import math
import numpy as np
import cv2

from . import aim_kin
from .geometry import *
from .utils import *

# aivision currently uses 320x240 dimensions but image is 640x480
AIVISION_RESOLUTION_SCALE = 2

class WorldObject():
    def __init__(self, id=None, name=None, x=0, y=0, z=0, theta=None, is_visible=False):
        self.id = id
        self.pose = Pose(x, y, z, theta)
        self.name = name or self.__class__.__name__
        self.matched = None  # matching object from data association
        self.is_fixed = False   # True for walls and markers in predefined maps
        self.is_obstacle = True
        self.is_visible = is_visible
        self.is_missing = False
        self.is_valid = True
        self.held_by = None
        self.is_foreign = False
        if is_visible:
            self.pose_confidence = +1
        else:
            self.pose_confidence = -1

    def __repr__(self):
        if self.is_visible:
            vis = "visible"
        elif self.is_missing:
            vis = "missing"
        else:
            vis = "unseen"
        return f'<{self.id or self.name} {vis} at ({self.pose.x:.1f}, {self.pose.y:.1f})>'

    def update_matched_object(self):
        MIN_SENSOR_NOISE = 5
        sensor_noise = max(MIN_SENSOR_NOISE, self.sensor_distance * 0.1)
        self.matched.pose.update(self.pose, sensor_noise)
        if hasattr(self, 'spec'):
            self.matched.spec = self.spec
        if hasattr(self, 'marker'):
            self.matched.marker = self.marker
        self.matched.is_visible = True
        self.matched.is_missing = False

class BarrelObj(WorldObject):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.name = spec['name']
        self.diameter = 22 # mm
        self.height = 25

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
        self.base_diameter = 22 # mm
        self.width = 38 # mm

    def __repr__(self):
        vis = "visible" if self.is_visible else "unseen"
        return f'<{self.id or self.name} {vis} at ({self.pose.x:.1f}, {self.pose.y:.1f}) @ {self.pose.theta*180/pi:.1f} deg.>'
    

class ArucoMarkerObj(WorldObject):
    def __init__(self, spec, x=0, y=0, z=0, theta=0):
        super().__init__(x, y, z, theta)
        self.name = spec['name']
        self.marker_id = spec['id']
        self.marker = spec['marker']
        self.pose_confidence = +1

    def __repr__(self):
        if self.pose_confidence >= 0:
            vis = ' visible' if self.is_visible else ''
            fix = ' fixed' if self.is_fixed else ''
            return '<ArucoMarkerObj %s: (%.1f, %.1f, %.1f) @ %d deg.%s%s>' % \
                (self.id, self.pose.x, self.pose.y, self.pose.z, self.pose.theta*180/pi, fix, vis)
        else:
            return f'<ArucoMarkerObj {self.id[12:]}: position unknown>'
        

class WallObj(WorldObject):
    def __init__(self, wall_spec, x=0, y=0, z=0, theta=0):
        super().__init__(x=x, y=y, z=z, theta=theta)
        self.wall_spec = wall_spec
        self.name = wall_spec.label
        self.length = wall_spec.length
        self.height = wall_spec.height
        self.is_fixed = True

    def __repr__(self):
        vis = 'visible' if self.is_visible else 'unseen'
        return f'<WallObj {self.name} ({self.pose.x:.1f}, {self.pose.y:.1f}) @ {self.pose.theta*180/pi:.1f} deg. {vis}>'

wall_marker_dict = dict()

class WallSpec():
    def __init__(self, label=None, length=100, height=210, marker_specs=dict(), doorways=dict()):
        self.length = length
        self.height = height
        self.marker_specs = marker_specs
        self.doorways = doorways
        marker_id_numbers = list(marker_specs.keys())
        self.label = label or f'Wall-{min(marker_id_numbers)}'
        global wall_marker_dict
        for id in marker_id_numbers:
            wall_marker_dict[id] = self
        wall_marker_dict[self.label] = self


class DoorwayObj(WorldObject):
    def __init__(self, wall, index):
        name = f'DoorwayObj-{wall.name[5:]}:{index}'
        super().__init__(name=name, is_visible=wall.is_visible)
        door_spec = wall.wall_spec.doorways[index]
        self.door_width = door_spec['width']
        self.wall = wall
        self.index = index  # which doorway is this?  0, 1, ...
        self.is_obstacle = False
        self.update()

    def update(self):
        door_spec = self.wall.wall_spec.doorways[self.index]
        self.pose = copy.deepcopy(self.wall.pose)
        self.sensor_distance = self.wall.sensor_distance

    def __repr__(self):
        if self.pose_confidence >= 0:
            return '<DoorwayObj %s: (%.1f,%.1f) @ %d deg.>' % \
                (self.id, self.pose.x, self.pose.y, self.pose.theta*180/pi)
        else:
            return '<DoorwayObj %s: position unknown>' % self.id

################################################################

class WorldMap():

    def __init__(self,robot):
        self.robot = robot
        self.objects = dict()
        self.pending_objects = dict()
        self.missing_objects = []
        self.shared_objects = dict()
        self.name_counts = dict()  # For generating new object names

    def __repr__(self):
        return f'<WorldMap with {len(self.objects)} objects>'

    def clear(self):
        self.objects.clear()
        self.pending_objects.clear()
        self.missing_objects = []
        self.shared_objects.clear()
        self.name_counts.clear()
        

    def update(self):
        #print('worldmap update')
        self.updated_objects = []
        self.make_new_objects_from_vision()
        self.associate_objects()
        self.update_associated_objects()
        self.detect_missing_objects()
        self.process_unassociated_objects()
        self.update_visibilities()
        # Consider deletion of unmatched worldmap objects
        # Update robot's pose if we have landmarks available

    def make_new_objects_from_vision(self):
        self.candidates = list()
        self.make_new_aiv_objects()
        if self.robot.aruco_detector:
            self.make_new_wall_objects()
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
            obj.is_visible = True
            # Calculate midpoint of bottom edge, which we assume is on the floor
            cx = (spec['originx'] + spec['width']/2) * AIVISION_RESOLUTION_SCALE
            # correct height for possible occlusion by foreground object
            corr_height = max(spec['height'], spec['width']*1.10)
            cy = (spec['originy'] + corr_height) * AIVISION_RESOLUTION_SCALE
            if isinstance(obj, AprilTagObj):
                cy += spec['height'] * 2 * AIVISION_RESOLUTION_SCALE
            hit = self.robot.kine.project_to_ground(cx, cy)
            # offset hit by half the object thickness
            if obj.__dict__.get('diameter'):
                hit += point(obj.diameter / 2, 0, 0)   # *** should calculate y offset too
            # convert to world coordinates
            robotpos = point(self.robot.pose.x, self.robot.pose.y)
            objpos = aboutZ(self.robot.pose.theta).dot(hit) + robotpos
            x = objpos[0][0]
            y = objpos[1][0]
            distance = ((x - self.robot.pose.x)**2 + (y - self.robot.pose.y)**2) ** 0.5
            MAX_DISTANCE = 300 # anything further than this is a spurious detection
            if distance > MAX_DISTANCE:
                continue
            obj.sensor_distance = distance
            if isinstance(obj, AprilTagObj):
                tag_angle_correction_factor = 4  # guesstimate
                angle = spec['angle'] - (0 if spec['angle'] < 180 else 360)
                theta = self.robot.pose.theta - angle / 180 * pi * tag_angle_correction_factor
            else:
                theta = None
            obj.pose = Pose(x, y, 0, theta)
            self.candidates.append(obj)

    def make_new_wall_objects(self):
        seen = self.robot.aruco_detector.seen_marker_objects.copy()
        wall_markers = dict()
        for (id,marker) in seen.items():
            if id in wall_marker_dict:
                spec = wall_marker_dict[id]
                if spec.label not in wall_markers:
                    wall_markers[spec.label] = list()
                wall_markers[spec.label].append((id,marker))
        for (wall_id,markers) in wall_markers.items():
            wall = self.infer_wall_from_corners_lists(wall_id, markers)
            self.candidates.append(wall)
            self.make_doorways_from_wall(wall)

    def infer_wall_from_corners_lists(self, wall_id, markers):
        # All these markers have the same wall_spec, so just grab the first one.
        wall_spec = wall_marker_dict[wall_id]
        marker_size = self.robot.aruco_detector.marker_size
        world_points = []
        image_points = []
        for (id, marker) in markers:
            s = wall_spec.marker_specs[id]['side']
            cx = wall_spec.marker_specs[id]['x']
            cy = wall_spec.marker_specs[id]['y']
            world_points.append((cx-s*marker_size/2, cy+marker_size/2, s))
            world_points.append((cx+s*marker_size/2, cy+marker_size/2, s))
            world_points.append((cx+s*marker_size/2, cy-marker_size/2, s))
            world_points.append((cx-s*marker_size/2, cy-marker_size/2, s))

            corners = marker.bbox[0]
            image_points.append(corners[0])
            image_points.append(corners[1])
            image_points.append(corners[2])
            image_points.append(corners[3])

        # Find rotation and translation vector from camera frame using SolvePnP
        (success, rvecs, tvecs) = cv2.solvePnP(np.array(world_points),
                                               np.array(image_points),
                                               self.robot.aruco_detector.camera_matrix,
                                               self.robot.aruco_detector.distortion_array)
        rotationm, jcob = cv2.Rodrigues(rvecs)
        # Change to marker frame.
        # Arucos seen head-on have orientation 0, so work with that for now.
        # Later we will flip the orientation to pi for the worldmap.
        transformed = np.matrix(rotationm).T*(-np.matrix(tvecs))
        angles_xyz = rotation_matrix_to_euler_angles(rotationm)
        # euler angle flip when back of wall is seen
        if angles_xyz[2] > pi/2:
            wall_orient = wrap_angle(-(angles_xyz[1]-pi))
        elif angles_xyz[2] >= -pi/2 and angles_xyz[2] <= pi/2:
            wall_orient = wrap_angle((angles_xyz[1]))
        else:
            wall_orient = wrap_angle(-(angles_xyz[1]+pi))

        wall_x = -transformed[2]*cos(wall_orient) + (transformed[0]-wall_spec.length/2)*sin(wall_orient)
        wall_y = (transformed[0]-wall_spec.length/2)*cos(wall_orient) - -transformed[2]*sin(wall_orient)
        #print('# markers=', len(markers), '   wall_x=', wall_x, '  wall_y=', wall_y)
        #import pdb
        #breakpoint()
        # Flip wall orientation to match ArUcos for worldmap
        wm_wall_orient = wrap_angle(pi - wall_orient)
        rel_coords = aboutZ(self.robot.pose.theta).dot(point(wall_x[0,0], wall_y[0,0]))
        x = self.robot.pose.x + rel_coords[0,0]
        y = self.robot.pose.y + rel_coords[1,0]
        wall = WallObj(wall_spec, x=x, y=y, theta=wrap_angle(self.robot.pose.theta + wm_wall_orient))
        wall.sensor_distance = math.sqrt(wall_x[0,0]**2 + wall_y[0,0]**2)
        wall.is_visible = True
        return wall

    def make_doorways_from_wall(self, wall):
        for (index, door_spec) in wall.wall_spec.doorways.items():
            door = DoorwayObj(wall, index)
            self.candidates.append(door)

    def generate_doorway_list(self):
        "Used by path-planner.py"
        doorways = []
        for (key,obj) in self.objects.items():
            if isinstance(obj,DoorwayObj):
                w = obj.door_width / 2
                doorway_threshold_theta = obj.pose.theta + pi/2
                dx = w * cos(doorway_threshold_theta)
                dy = w * sin(doorway_threshold_theta)
                ox = obj.pose.x
                oy = obj.pose.y
                doorways.append((obj, ((ox-dx, oy-dy), (ox+dx, oy+dy))))
        return doorways

    def make_new_aruco_objects(self):
        for (id,marker) in self.robot.aruco_detector.seen_marker_objects.items():
            #if id in wall_marker_dict:
            #   continue
            name = f'ArucoMarker-{id}'
            spec = {'name': name, 'id': id, 'marker': marker}
            sensor_dist = marker.camera_distance
            camera_offset = np.array([0, 0, aim_kin.camera_from_origin])
            sensor_coords = marker.camera_coords + camera_offset
            sensor_bearing = atan2(sensor_coords[0], sensor_coords[2])
            sensor_orient = wrap_angle(pi - marker.euler_rotation[1] * (pi/180))
            theta = self.robot.pose.theta
            #print(f'sdist={sensor_dist} scoords={sensor_coords} sbearing={sensor_bearing*180/pi} sorient={sensor_orient*180/pi} theta={theta*180/pi}')
            obj = self.make_object(spec)
            obj.pose = Pose(self.robot.pose.x + sensor_dist * cos(theta + sensor_bearing),
                            self.robot.pose.y + sensor_dist * sin(theta + sensor_bearing),
                            marker.aruco_parent.marker_size / 2,  # *** TEMPORARY HACK ***
                            wrap_angle(self.robot.pose.theta - sensor_orient))
            obj.sensor_distance = sensor_dist
            obj.is_visible = True
            self.candidates.append(obj)

    def associate_objects(self):
        obj_types = list(set(type(obj) for obj in self.candidates))
        for otype in obj_types:
            self.associate_objects_of_type(otype)

    def association_cost(self, obj1, obj2):
        cost = ((obj1.pose.x-obj2.pose.x)**2 + (obj1.pose.y-obj2.pose.y)**2)
        return cost

    def associate_objects_of_type(self, otype):
        new = [c for c in self.candidates if type(c) is otype]
        old = [o for o in self.objects.values() if type(o) is otype]
        N_new = len(new)
        N_old = len(old)
        if N_old == 0:
            return
        costs = np.zeros([N_new,N_old])
        if self.robot.particle_filter and \
           self.robot.particle_filter.state in (self.robot.particle_filter.LOST,
                                                self.robot.particle_filter.LOCALIZING):
            MAX_ACCEPTABLE_COST = np.inf
        elif otype in (ArucoMarkerObj, WallObj):
            MAX_ACCEPTABLE_COST = 5000  # should adjust based on pf undertainty
        else:
            MAX_ACCEPTABLE_COST = 200  # should adjust based on pf undertainty
        for i in range(N_new):
            for j in range(N_old):
                if otype is ArucoMarkerObj and new[i].marker_id != old[j].marker_id:
                    costs[i,j] = MAX_ACCEPTABLE_COST + 1
                else:
                    costs[i,j] = self.association_cost(new[i], old[j])
        # *** Greedy algorithm; replace with the Hungarian algorithm
        for i in range(N_new):
            bestj = costs[i,:].argmin()
            if costs[i,bestj] < MAX_ACCEPTABLE_COST:
                new[i].matched = old[bestj]
                costs[:,bestj] = 1 + MAX_ACCEPTABLE_COST

    def update_associated_objects(self):
        for candidate in self.candidates:
            if candidate.matched:
                candidate.update_matched_object()
                self.updated_objects.append(candidate.matched)
                candidate.matched.is_missing = False
                if candidate.matched in self.missing_objects:
                    self.missing_objects.remove(candidate.matched)

    def should_be_visible(self, obj):
        # Really crude approach for now.  Should be doing camera
        # projection and accounting for occlusion.  In the future we
        # should employ the depth map for occusion detection.
        # For now, just return true if the object's bearing is within
        # the camera field of view and the distance is not too large.
        dx = obj.pose.x - self.robot.pose.x
        dy = obj.pose.y - self.robot.pose.y
        bearing = wrap_angle(atan2(dy,dx) - self.robot.pose.theta)
        distance = (dx**2 + dy**2) ** 0.5
        DISTANCE_THRESHOLD = 400 # mm
        BEARING_THRESHOLD = 30 # degrees
        result = abs(bearing)*180/pi < BEARING_THRESHOLD and distance < DISTANCE_THRESHOLD
        return result

    def detect_missing_objects(self):
        for obj in self.objects.values():
            if obj not in self.updated_objects and self.should_be_visible(obj):
                if obj not in self.missing_objects:
                    obj.is_visible = False
                    obj.is_missing = True
                    self.missing_objects.append(obj)
                    #print('missing object:', obj)

    def process_unassociated_objects(self):
        """
        The vision system produces lots of spurious objects, so we require
        a new object to be seen 6 times in successive camera frames before
        we add it to the world map.
        """
        unassociated = [c for c in self.candidates if c.matched is None]
        pending = list(self.pending_objects.keys())
        COST_THRESHOLD = 50
        if self.robot.particle_filter and \
           self.robot.particle_filter.state in (self.robot.particle_filter.LOST,
                                                self.robot.particle_filter.LOCALIZING):
            if unassociated:
                pass # print("Not localized: can't add", unassociated)
            return
        if self.robot.particle_filter:
            pass # print('robot.particle_filter.state=', self.robot.particle_filter.state)
        for candidate in unassociated:
            matches = [p for p in pending if self.association_cost(candidate,p) < COST_THRESHOLD]
            if matches:
                m = matches[0]
                self.pending_objects[m] += 1
                if self.pending_objects[m] >= 6:
                    if self.reclaim_object(candidate):
                        pass
                    else:
                        candidate.id = self.next_in_sequence(candidate.name)
                        candidate.pose = PoseEstimate(candidate.pose)
                        self.objects[candidate.id] = candidate
                        candidate.is_visible = True
                        print('Added', candidate)
                        self.updated_objects.append(candidate)
                    del self.pending_objects[m]
                pending.remove(m)
            else:
                self.pending_objects[candidate] = 1
                #print('proposed', candidate)
        for p in pending:
            #print('retracted', p, '  count=', self.pending_objects[p])
            del self.pending_objects[p]

    def reclaim_object(self, obj):
        t = type(obj)
        missing = [m for m in self.missing_objects if type(m) == t]
        if hasattr(obj,'marker_id'):
            missing = [m for m in missing if m.marker_id == obj.marker_id]
        if len(missing) == 0:
            return None
        costs = [self.association_cost(obj, m) for m in missing]
        min_index = np.argmin(costs)
        match = missing[min_index]
        match.is_visible = True
        match.pose = PoseEstimate(obj.pose)
        self.updated_objects.append(match)
        self.missing_objects.remove(match)
        match.is_visible = True
        #print('reclaimed', match)
        return match
        
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
            if obj not in self.updated_objects:
                obj.is_visible = False

    def show_objects(self):
        objs = sorted(self.objects.items(), key=lambda x: x[0])
        if len(objs) == 0:
            print('No objects in the world map.\n')
            return
        width = max([len(x[0]) for x in objs])
        for obj in objs:
            print(f'{obj[0].rjust(width)}: {obj[1]}')
        print()



################ GPT interface ################

    def get_prompt(self):
        prompt = ''
        prompt += f'You are located at ({round(self.robot.pose.x)}, {round(self.robot.pose.y)})\n'
        prompt += f'Your heading is {round(self.robot.pose.theta*180/pi)} degrees\n'
        prompt += f'Your battery level is {self.robot.battery_percentage} percent.\n'
        for (id,obj) in self.objects.items():
            if not obj.is_missing:
                if obj.is_visible:
                    vis = "visible"
                else:
                    vis = "not vislbie"
                prompt += f'{id} is located at ({round(obj.pose.x)}, {round(obj.pose.y)}) ' + \
                    f'and is {vis}\n'
            else:
                prompt += f'{id} is missing\n'
        return prompt
