from numpy import matrix, tan, arctan2
from math import sin, cos, atan2, pi, sqrt

from .utils import *
from .nodes import *
from .transitions import *
from .geometry import *
from .pilot0 import *
from .worldmap import WallObj, DoorwayObj

class DoorPass(StateNode):
    """Pass through a doorway. Assumes the doorway is nearby and unobstructed."""

    OUTER_GATE_DISTANCE = 150 # mm
    INNER_GATE_DISTANCE =  70 # mm

    def __init__(self, door_spec=None):
        self.door_spec = door_spec
        super().__init__()

    def start(self, event=None):
        door_spec = self.door_spec
        if isinstance(event,DataEvent):
            door_spec = event.data
        if isinstance(door_spec, int):
            door_spec ='Doorway-%d:0.a' % door_spec
        if isinstance(door_spec, str):
            doorobj = self.robot.world_map.objects.get(door_spec)
        elif isinstance(door_spec, DoorwayObj):
            doorobj = door_spec
        else:
            doorobj = None
        if isinstance(doorobj, DoorwayObj):
            self.object = doorobj
        else:
            print("Error in DoorPass: no doorway named %s" % repr(door_spec))
            raise ValueError(door_spec, doorobj)
        super().start(event)


    @staticmethod
    def calculate_gate(start_point, door, offset=0):
        """Returns closest gate point (gx, gy)"""
        (rx,ry) = start_point
        dx = door.pose.x
        dy = door.pose.y
        dtheta = door.pose.theta
        # calculate gates on either side of the door, then pick the closest
        pt1x = dx + offset * cos(dtheta)
        pt1y = dy + offset * sin(dtheta)
        pt2x = dx + offset * cos(dtheta+pi)
        pt2y = dy + offset * sin(dtheta+pi)
        dist1sq = (pt1x-rx)**2 + (pt1y-ry)**2
        dist2sq = (pt2x-rx)**2 + (pt2y-ry)**2
        if dist1sq < dist2sq:
            return (pt1x, pt1y, wrap_angle(dtheta+pi))
        else:
            return (pt2x, pt2y, dtheta)

    class AwayFromCollide(Forward):
        def start(self, event=None):
            if isinstance(event,DataEvent):
                startNode = event.data[0]
                collideObj = event.data[1]
                pose = self.robot.pose
                (rx, ry, rtheta) = pose.x, pose.y, pose.theta
                cx, cy = collideObj.center[0,0],collideObj.center[1,0]
                ctheta = atan2(cy-ry, cx-rx)
                delta_angle = wrap_angle(ctheta - rtheta)
                delta_angle = delta_angle/pi*180
                if -90 < delta_angle and delta_angle < 90:
                    self.distance_mm = -40
                else:
                    self.distance_mm = 40
                self.drive_speed = 50
                super().start(event)
            else:
                raise ValueError('DataEvent to AwayFromCollide must be a StartCollides.args', event.data)
                super().start(event)
                self.post_failure()


    class TurnToGate(Turn):
        """Turn to the approach gate, or post success if we're already there."""
        def __init__(self,offset):
            self.offset = offset
            super().__init__(turn_speed=45)

        def start(self,event=None):
            pose = self.robot.pose
            (rx, ry, rtheta) = pose.x, pose.y, pose.theta
            (gate_x, gate_y, _) = DoorPass.calculate_gate((rx,ry), self.parent.object, offset=self.offset)
            bearing = atan2(gate_y-ry, gate_x-rx)
            turn = wrap_angle(bearing - rtheta)
            print('^^ TurnToGate: gate=(%.1f, %.1f)  offset=%.1f rtheta=%.1f  bearing=%.1f  turn=%.1f' %
                  (gate_x, gate_y, self.offset, rtheta*180/pi, bearing*180/pi, turn*180/pi))
            MIN_TURN_ANGLE = 2.5 * (pi/180)
            if abs(turn) < MIN_TURN_ANGLE:
                self.angle_deg = 0
                super().start(event)
                self.post_completion()
            else:
                self.angle_deg = turn * 180/pi
                super().start(event)


    class ForwardToGate(Forward):
        """Travel forward to reach the approach gate."""
        def __init__(self,offset):
            self.offset = offset
            super().__init__()

        def start(self,event=None):
            rx = self.robot.pose.x
            ry = self.robot.pose.y
            (gate_x, gate_y, _) = DoorPass.calculate_gate((rx,ry), self.parent.object, offset=self.offset)
            self.distance_mm = sqrt((gate_x-rx)**2 + (gate_y-ry)**2)
            self.drive_speed = 50
            super().start(event)

    class TurnToFaceWall(Turn):
        def start(self,event=None):
            door_theta = self.parent.object.pose.theta
            robot_theta = self.robot.pose.theta
            if abs(wrap_angle(door_theta - robot_theta)) > 90:
                door_theta += pi
            self.angle_deg = wrap_angle(door_theta - robot_theta) * 180/pi
            super().start(event)

    class CenterOnDoorway(Sideways):
        def start(self,event=None):
            """Calculate the vector from the robot to the doorway and rotate it into the robot's
            frame so the y component gives the sideways distance to line up with the door."""
            # NOTE: since the robot won't be perfectly facing the door we should perhaps
            # adjust theta to reflect that.
            door_pose = self.parent.object.pose
            robot_pose = self.robot.pose
            dx = door_pose.x - robot_pose.x
            dy = door_pose.y - robot_pose.y
            vec = aboutZ(robot_pose.theta).dot(point(dx,dy))
            self.distance_mm = vec[1,0]
            print(f'dpose={door_pose} rpose={robot_pose} dx={neaten(dx)} dy={neaten(dy)}' +
                  f' vec={neaten(vec[0,0])},{neaten(vec[1,0])} dist={neaten(self.distance_mm)}')
            super().start(event)
        

    class TurnToMarker(Turn):
        """Use camera image and native pose to center the door marker."""
        def start(self,event=None):
            marker_ids = self.parent.object.marker_ids
            marker = self.robot.aruco_detector.seen_marker_objects.get(marker_ids[0], None) or \
                     self.robot.aruco_detector.seen_marker_objects.get(marker_ids[1], None)
            if not marker:
                self.angle_deg = 0
                super().start(event)
                print("TurnToMarker failed to find marker %s or %s!" % marker_ids)
                self.post_failure()
                return
            else:
                print('TurnToMarker saw marker', marker)
            sensor_dist = marker.camera_distance
            sensor_bearing = atan2(marker.camera_coords[0],
                                   marker.camera_coords[2])
            x = self.robot.pose.position.x
            y = self.robot.pose.position.y
            theta = self.robot.pose.rotation.angle_z.radians
            direction = theta + sensor_bearing
            dx = sensor_dist * cos(direction)
            dy = sensor_dist * sin(direction)
            turn = wrap_angle(atan2(dy,dx) - self.robot.pose.rotation.angle_z.radians)
            if abs(turn) < 0.5*pi/180:
                self.angle_deg = 0
            else:
                self.angle_deg = turn * 180/pi
            print("TurnToMarker %s turning by %.1f degrees" % (self.name, self.angle.degrees))
            super().start(event)

    class DriveThrough(Forward):
        """Travel forward to drive through the gate."""
        def __init__(self):
            super().__init__()

        def start(self,event=None):
            pose = self.robot.pose
            (rx, ry, rtheta) = pose.x, pose.y, pose.theta
            (gate_x, gate_y, gate_theta) = DoorPass.calculate_gate((rx,ry), self.parent.object, offset=5)
            dist = sqrt((gate_x-rx)**2 + (gate_y-ry)**2)
            offset = 120
            delta_theta = wrap_angle(rtheta-(gate_theta+pi/2))
            delta_dist = abs(offset/sin(delta_theta))
            dist += delta_dist
            self.distance_mm = dist
            self.drive_speed = 50
            super().start(event)

    def setup(self):
        #         check_start: PilotCheckStartDetail()
        #         check_start =S=> turn_to_gate1
        #         check_start =D=> away_from_collide
        #         check_start =F=> Forward(-80) =C=> check_start2
        # 
        #         turn_to_gate1: self.TurnToGate(DoorPass.OUTER_GATE_DISTANCE) =C=>
        #             Print('turned toward outer gate') =T(0.2)=> forward_to_gate1
        # 
        #         forward_to_gate1: self.ForwardToGate(DoorPass.OUTER_GATE_DISTANCE) =C=>
        #             Print('arrived at outer gate') =T(0.2)=> turn_to_gate2
        # 
        #         turn_to_gate2: self.TurnToGate(DoorPass.INNER_GATE_DISTANCE) =C=>
        #             Print('turned to inner gate') =T(0.2)=> align_with_wall
        # 
        #         align_with_wall: self.TurnToFaceWall() =C=>
        #             Print('facing the wall') =T(0.2)=> center_on_doorway
        # 
        #         center_on_doorway: self.CenterOnDoorway() =C=>
        #             Print('centered on doorway') =C=> through_door
        # 
        #         through_door: self.DriveThrough() =C=> ParentCompletes()
        # 
        #         away_from_collide: self.AwayFromCollide() =C=> StateNode() =T(0.2)=> check_start2
        #         away_from_collide =F=> check_start2
        # 
        #         check_start2: PilotCheckStartDetail()
        #         check_start2 =S=> turn_to_gate1
        #         check_start2 =D=> away_from_collide2
        #         check_start2 =F=> ParentFails()
        # 
        #         away_from_collide2: self.AwayFromCollide() =C=> StateNode() =T(0.2)=> check_start3
        #         away_from_collide2 =F=> check_start3
        # 
        #         check_start3: PilotCheckStart()
        #         check_start3 =S=> turn_to_gate1
        #         check_start3 =F=> ParentFails()
        # 
        
        # Code generated by genfsm on Mon Apr 14 13:34:26 2025:
        
        check_start = PilotCheckStartDetail() .set_name("check_start") .set_parent(self)
        forward1 = Forward(-80) .set_name("forward1") .set_parent(self)
        turn_to_gate1 = self.TurnToGate(DoorPass.OUTER_GATE_DISTANCE) .set_name("turn_to_gate1") .set_parent(self)
        print1 = Print('turned toward outer gate') .set_name("print1") .set_parent(self)
        forward_to_gate1 = self.ForwardToGate(DoorPass.OUTER_GATE_DISTANCE) .set_name("forward_to_gate1") .set_parent(self)
        print2 = Print('arrived at outer gate') .set_name("print2") .set_parent(self)
        turn_to_gate2 = self.TurnToGate(DoorPass.INNER_GATE_DISTANCE) .set_name("turn_to_gate2") .set_parent(self)
        print3 = Print('turned to inner gate') .set_name("print3") .set_parent(self)
        align_with_wall = self.TurnToFaceWall() .set_name("align_with_wall") .set_parent(self)
        print4 = Print('facing the wall') .set_name("print4") .set_parent(self)
        center_on_doorway = self.CenterOnDoorway() .set_name("center_on_doorway") .set_parent(self)
        print5 = Print('centered on doorway') .set_name("print5") .set_parent(self)
        through_door = self.DriveThrough() .set_name("through_door") .set_parent(self)
        parentcompletes1 = ParentCompletes() .set_name("parentcompletes1") .set_parent(self)
        away_from_collide = self.AwayFromCollide() .set_name("away_from_collide") .set_parent(self)
        statenode1 = StateNode() .set_name("statenode1") .set_parent(self)
        check_start2 = PilotCheckStartDetail() .set_name("check_start2") .set_parent(self)
        parentfails1 = ParentFails() .set_name("parentfails1") .set_parent(self)
        away_from_collide2 = self.AwayFromCollide() .set_name("away_from_collide2") .set_parent(self)
        statenode2 = StateNode() .set_name("statenode2") .set_parent(self)
        check_start3 = PilotCheckStart() .set_name("check_start3") .set_parent(self)
        parentfails2 = ParentFails() .set_name("parentfails2") .set_parent(self)
        
        successtrans1 = SuccessTrans() .set_name("successtrans1")
        successtrans1 .add_sources(check_start) .add_destinations(turn_to_gate1)
        
        datatrans1 = DataTrans() .set_name("datatrans1")
        datatrans1 .add_sources(check_start) .add_destinations(away_from_collide)
        
        failuretrans1 = FailureTrans() .set_name("failuretrans1")
        failuretrans1 .add_sources(check_start) .add_destinations(forward1)
        
        completiontrans1 = CompletionTrans() .set_name("completiontrans1")
        completiontrans1 .add_sources(forward1) .add_destinations(check_start2)
        
        completiontrans2 = CompletionTrans() .set_name("completiontrans2")
        completiontrans2 .add_sources(turn_to_gate1) .add_destinations(print1)
        
        timertrans1 = TimerTrans(0.2) .set_name("timertrans1")
        timertrans1 .add_sources(print1) .add_destinations(forward_to_gate1)
        
        completiontrans3 = CompletionTrans() .set_name("completiontrans3")
        completiontrans3 .add_sources(forward_to_gate1) .add_destinations(print2)
        
        timertrans2 = TimerTrans(0.2) .set_name("timertrans2")
        timertrans2 .add_sources(print2) .add_destinations(turn_to_gate2)
        
        completiontrans4 = CompletionTrans() .set_name("completiontrans4")
        completiontrans4 .add_sources(turn_to_gate2) .add_destinations(print3)
        
        timertrans3 = TimerTrans(0.2) .set_name("timertrans3")
        timertrans3 .add_sources(print3) .add_destinations(align_with_wall)
        
        completiontrans5 = CompletionTrans() .set_name("completiontrans5")
        completiontrans5 .add_sources(align_with_wall) .add_destinations(print4)
        
        timertrans4 = TimerTrans(0.2) .set_name("timertrans4")
        timertrans4 .add_sources(print4) .add_destinations(center_on_doorway)
        
        completiontrans6 = CompletionTrans() .set_name("completiontrans6")
        completiontrans6 .add_sources(center_on_doorway) .add_destinations(print5)
        
        completiontrans7 = CompletionTrans() .set_name("completiontrans7")
        completiontrans7 .add_sources(print5) .add_destinations(through_door)
        
        completiontrans8 = CompletionTrans() .set_name("completiontrans8")
        completiontrans8 .add_sources(through_door) .add_destinations(parentcompletes1)
        
        completiontrans9 = CompletionTrans() .set_name("completiontrans9")
        completiontrans9 .add_sources(away_from_collide) .add_destinations(statenode1)
        
        timertrans5 = TimerTrans(0.2) .set_name("timertrans5")
        timertrans5 .add_sources(statenode1) .add_destinations(check_start2)
        
        failuretrans2 = FailureTrans() .set_name("failuretrans2")
        failuretrans2 .add_sources(away_from_collide) .add_destinations(check_start2)
        
        successtrans2 = SuccessTrans() .set_name("successtrans2")
        successtrans2 .add_sources(check_start2) .add_destinations(turn_to_gate1)
        
        datatrans2 = DataTrans() .set_name("datatrans2")
        datatrans2 .add_sources(check_start2) .add_destinations(away_from_collide2)
        
        failuretrans3 = FailureTrans() .set_name("failuretrans3")
        failuretrans3 .add_sources(check_start2) .add_destinations(parentfails1)
        
        completiontrans10 = CompletionTrans() .set_name("completiontrans10")
        completiontrans10 .add_sources(away_from_collide2) .add_destinations(statenode2)
        
        timertrans6 = TimerTrans(0.2) .set_name("timertrans6")
        timertrans6 .add_sources(statenode2) .add_destinations(check_start3)
        
        failuretrans4 = FailureTrans() .set_name("failuretrans4")
        failuretrans4 .add_sources(away_from_collide2) .add_destinations(check_start3)
        
        successtrans3 = SuccessTrans() .set_name("successtrans3")
        successtrans3 .add_sources(check_start3) .add_destinations(turn_to_gate1)
        
        failuretrans5 = FailureTrans() .set_name("failuretrans5")
        failuretrans5 .add_sources(check_start3) .add_destinations(parentfails2)
        
        return self
