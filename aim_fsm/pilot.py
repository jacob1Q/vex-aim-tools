import math
import time
import sys
import asyncio

from .base import *
from .utils import Pose
from .rrt import *
from .nodes import *
from .events import PilotEvent
from .transitions import *
from .worldmap import WorldObject  #, DoorwayObj
from .path_planner import PathPlanner, PathPlannerNode #, PathPlannerProcess
from .geometry import segment_intersect_test
#from .doorpass import DoorPass
from .pilot0 import *

class DoorPass(StateNode): pass   # dummy

#---------------- Pilot Exceptions and Events ----------------

class PilotException(Exception):
    def __str__(self):
        return self.__repr__()

class InvalidPose(PilotException): pass
class CollisionDetected(PilotException): pass

# Note: StartCollides, GoalCollides, and MaxIterations exceptions are defined in rrt.py.

class ParentPilotEvent(StateNode):
    """Receive a PilotEvent and repost it from the receiver's parent. This allows
     derived classes that use the Pilot to make its PilotEvents visible."""
    def start(self,event):
        super().start(event)
        if not isinstance(event,PilotEvent):
            raise TypeError("ParentPilotEvent must be invoked with a PilotEvent, not %s" % event)
        if 'grid_display' in event.args:
            self.robot.rrt.grid_display = event.args['grid_display']
        event2 = PilotEvent(event.status)
        event2.args = event.args
        self.parent.post_event(event2)

#---------------- PilotBase ----------------

class PilotBase(StateNode):

    """Base class for PilotToObject, PilotToPose, etc."""

    class ClearDisplays(StateNode):
        def start(self,event=None):
            super().start()
            if self.robot.path_viewer:
                self.robot.path_viewer.clear()

    class SendObject(StateNode):
        def start(self,event=None):
          super().start()
          object = self.parent.object
          #if object.pose_confidence < 0:
          #   self.parent.post_event(PilotEvent(NotLocalized,object=object))
          #   self.parent.post_failure()
          #   return
          self.post_event(DataEvent(self.parent.object))

    class ReceivePlan(StateNode):
        def start(self, event=None):
            super().start(event)
            if not isinstance(event, DataEvent):
                raise ValueError(event)
            (navplan, grid_display) = event.data

            self.robot.rrt.draw_path = navplan.extract_path()
            #print('ReceivePlan: draw_path=', self.robot.world.rrt.draw_path)
            self.robot.rrt.grid_display = grid_display
            self.post_event(DataEvent(navplan))

    class PilotExecutePlan(StateNode):
        def start(self, event=None):
            if not isinstance(event, DataEvent) and isinstance(event.data, NavPlan):
                raise ValueError(event)
            self.index = 0
            if self.parent.execute:
                self.navplan = event.data
            else:
                self.parent.parent.post_completion()
                return
            super().start(event)

        class DispatchStep(StateNode):
            def start(self, event=None):
                super().start(event)
                step = self.parent.navplan.steps[self.parent.index]
                print('nav step', step)
                self.post_event(DataEvent(step.type))

        class ExecuteDrive(Forward):
            def start(self, event=None):
                step = self.parent.navplan.steps[self.parent.index]
                print('step=',step)
                super().start(event)

        class ExecuteDoorPass(DoorPass):
            def start(self, event=None):
                step = self.parent.navplan.steps[self.parent.index]
                super().start(DataEvent(step.param))

        class ExecuteBackup(Forward):
            def start(self, event=None):
                step = self.parent.navplan.steps[self.parent.index]
                if len(step.param) > 1:
                    print('***** WARNING: extra backup steps not being processed *****')
                node = step.param[0]
                dx = node.x - self.robot.pose.x
                dy = node.y - self.robot.pose.y
                self.distance_mm = - sqrt(dx*dx + dy*dy)
                super().start(event)

        class NextStep(StateNode):
            def start(self, event=None):
                super().start(event)
                self.parent.index += 1
                if self.parent.index < len(self.parent.navplan.steps):
                    self.post_success()
                else:
                    self.post_completion()

        def setup(self):
            #             # PilotExecutePlan machine
            # 
            #             dispatch: self.DispatchStep()
            #             dispatch =D(NavStep.DRIVE)=> drive
            #             dispatch =D(NavStep.DOORPASS)=> doorpass
            #             dispatch =D(NavStep.BACKUP)=> backup
            # 
            #             drive: self.ExecuteDrive()
            #             drive =C=> next
            #             drive =F=> ParentFails()
            # 
            #             doorpass: self.ExecuteDoorPass()
            #             doorpass =C=> next
            #             doorpass =F=> ParentFails()
            # 
            #             backup: self.ExecuteBackup()
            #             backup =C=> next
            #             backup =F=> ParentFails()
            # 
            #             next: self.NextStep()
            #             next =S=> dispatch
            #             next =C=> ParentCompletes()
            
            # Code generated by genfsm on Fri Feb 21 11:00:40 2025:
            
            dispatch = self.DispatchStep() .set_name("dispatch") .set_parent(self)
            drive = self.ExecuteDrive() .set_name("drive") .set_parent(self)
            parentfails1 = ParentFails() .set_name("parentfails1") .set_parent(self)
            doorpass = self.ExecuteDoorPass() .set_name("doorpass") .set_parent(self)
            parentfails2 = ParentFails() .set_name("parentfails2") .set_parent(self)
            backup = self.ExecuteBackup() .set_name("backup") .set_parent(self)
            parentfails3 = ParentFails() .set_name("parentfails3") .set_parent(self)
            next = self.NextStep() .set_name("next") .set_parent(self)
            parentcompletes1 = ParentCompletes() .set_name("parentcompletes1") .set_parent(self)
            
            datatrans1 = DataTrans(NavStep.DRIVE) .set_name("datatrans1")
            datatrans1 .add_sources(dispatch) .add_destinations(drive)
            
            datatrans2 = DataTrans(NavStep.DOORPASS) .set_name("datatrans2")
            datatrans2 .add_sources(dispatch) .add_destinations(doorpass)
            
            datatrans3 = DataTrans(NavStep.BACKUP) .set_name("datatrans3")
            datatrans3 .add_sources(dispatch) .add_destinations(backup)
            
            completiontrans1 = CompletionTrans() .set_name("completiontrans1")
            completiontrans1 .add_sources(drive) .add_destinations(next)
            
            failuretrans1 = FailureTrans() .set_name("failuretrans1")
            failuretrans1 .add_sources(drive) .add_destinations(parentfails1)
            
            completiontrans2 = CompletionTrans() .set_name("completiontrans2")
            completiontrans2 .add_sources(doorpass) .add_destinations(next)
            
            failuretrans2 = FailureTrans() .set_name("failuretrans2")
            failuretrans2 .add_sources(doorpass) .add_destinations(parentfails2)
            
            completiontrans3 = CompletionTrans() .set_name("completiontrans3")
            completiontrans3 .add_sources(backup) .add_destinations(next)
            
            failuretrans3 = FailureTrans() .set_name("failuretrans3")
            failuretrans3 .add_sources(backup) .add_destinations(parentfails3)
            
            successtrans1 = SuccessTrans() .set_name("successtrans1")
            successtrans1 .add_sources(next) .add_destinations(dispatch)
            
            completiontrans4 = CompletionTrans() .set_name("completiontrans4")
            completiontrans4 .add_sources(next) .add_destinations(parentcompletes1)
            
            return self

        # End of PilotExecutePlan
    # End of PilotBase


#---------------- PilotToObject ----------------

class PilotToObject(PilotBase):
    "Use the wavefront planner to navigate to a distant object."
    def __init__(self, object=None):
        super().__init__()
        self.object=object

    def start(self, event=None):
        if isinstance(event,DataEvent):
            if isinstance(event.data, WorldObject):
                self.object = event.data
            else:
                raise ValueError('DataEvent to PilotToObject must be a WorldObject', event.data)
        super().start(event)

    class CheckArrival(StateNode):
        def start(self, event=None):
            super().start(event)
            pf_pose = self.robot.pose
            if True: # *** TODO: check if we've arrived at the target shape
                self.post_success()
            else:
                self.post_failure()

    def setup(self):
        #         # PilotToObject machine
        # 
        #         launch: self.ClearDisplays() =N=> self.SendObject() =D=> planner
        # 
        #         #planner: PathPlannerProcess() =D=> recv
        #         planner: PathPlannerNode() =D=> recv
        #         planner =PILOT=> ParentPilotEvent() =N=> Print('Path planner failed')
        # 
        #         recv: self.ReceivePlan() =D=> exec
        # 
        #         exec: self.PilotExecutePlan()
        #         exec =C=> check
        #         exec =F=> ParentFails()
        # 
        #         check: self.CheckArrival()
        #         check =S=> ParentCompletes()
        #         check =F=> planner
        
        # Code generated by genfsm on Fri Feb 21 11:00:40 2025:
        
        launch = self.ClearDisplays() .set_name("launch") .set_parent(self)
        sendobject1 = self.SendObject() .set_name("sendobject1") .set_parent(self)
        planner = PathPlannerNode() .set_name("planner") .set_parent(self)
        parentpilotevent1 = ParentPilotEvent() .set_name("parentpilotevent1") .set_parent(self)
        print1 = Print('Path planner failed') .set_name("print1") .set_parent(self)
        recv = self.ReceivePlan() .set_name("recv") .set_parent(self)
        exec = self.PilotExecutePlan() .set_name("exec") .set_parent(self)
        parentfails4 = ParentFails() .set_name("parentfails4") .set_parent(self)
        check = self.CheckArrival() .set_name("check") .set_parent(self)
        parentcompletes2 = ParentCompletes() .set_name("parentcompletes2") .set_parent(self)
        
        nulltrans1 = NullTrans() .set_name("nulltrans1")
        nulltrans1 .add_sources(launch) .add_destinations(sendobject1)
        
        datatrans4 = DataTrans() .set_name("datatrans4")
        datatrans4 .add_sources(sendobject1) .add_destinations(planner)
        
        datatrans5 = DataTrans() .set_name("datatrans5")
        datatrans5 .add_sources(planner) .add_destinations(recv)
        
        pilottrans1 = PilotTrans() .set_name("pilottrans1")
        pilottrans1 .add_sources(planner) .add_destinations(parentpilotevent1)
        
        nulltrans2 = NullTrans() .set_name("nulltrans2")
        nulltrans2 .add_sources(parentpilotevent1) .add_destinations(print1)
        
        datatrans6 = DataTrans() .set_name("datatrans6")
        datatrans6 .add_sources(recv) .add_destinations(exec)
        
        completiontrans5 = CompletionTrans() .set_name("completiontrans5")
        completiontrans5 .add_sources(exec) .add_destinations(check)
        
        failuretrans4 = FailureTrans() .set_name("failuretrans4")
        failuretrans4 .add_sources(exec) .add_destinations(parentfails4)
        
        successtrans2 = SuccessTrans() .set_name("successtrans2")
        successtrans2 .add_sources(check) .add_destinations(parentcompletes2)
        
        failuretrans5 = FailureTrans() .set_name("failuretrans5")
        failuretrans5 .add_sources(check) .add_destinations(planner)
        
        return self

#---------------- PilotToPose ----------------

class PilotToPose(PilotBase):
    "Use the rrt path planner for short-range navigation to a specific pose."
    def __init__(self, target_pose=None, verbose=False, max_iter=RRT.DEFAULT_MAX_ITER, execute=True):
        super().__init__()
        self.target_pose = target_pose
        self.verbose = verbose
        self.max_iter = max_iter
        self.execute = execute

    def start(self, event=None):
        print('event=',event, event.data)
        if isinstance(event, DataEvent):
            if isinstance(event.data, WorldObject):
                self.target_object = event.data
                self.target_pose = event.data.pose
            elif isinstance(event.data, Pose):
                self.target_object = None
                self.target_pose = event.data
            else:
                raise ValueError(event.data)
            self.robot.rrt.max_iter = self.max_iter
        super().start(event)

    class PilotRRTPlanner(StateNode):
        def planner(self,start_node,goal_node):
            return self.robot.rrt.plan_path(start_node,goal_node, goal_object=self.parent.target_object)

        def start(self,event=None):
            super().start(event)
            tpose = self.parent.target_pose
            if tpose is None or (tpose.x == 0 and tpose.y == 0 and tpose.theta == 0):
                print("Pilot: target pose is invalid: %s" % tpose)
                self.parent.post_event(PilotEvent(InvalidPose, pose=tpose))
                self.parent.post_failure()
                return
            pose = self.robot.pose
            start_node = RRTNode(x=pose.x, y=pose.y, q=pose.theta)
            goal_node = RRTNode(x=tpose.x, y=tpose.y, q=tpose.theta)

            start_escape_move = None
            try:
                (treeA, treeB, path) = self.planner(start_node, goal_node)

            except StartCollides as e:
                # See if we can escape the start collision using canned headings.
                # This could be made more sophisticated, e.g., using arcs.
                #print('planner',e,'start',start_node)
                escape_distance = 50 # mm
                escape_headings = (0, +30/180.0*pi, -30/180.0*pi, pi, pi/2, -pi/2)
                for phi in escape_headings:
                    if phi != pi:
                        new_q = wrap_angle(start_node.q + phi)
                        d = escape_distance
                    else:
                        new_q = start_node.q
                        d = -escape_distance
                    new_start = RRTNode(x=start_node.x + d*cos(new_q),
                                        y=start_node.y + d*sin(new_q),
                                        q=new_q)
                    # print('trying start escape', new_start)
                    if not self.robot.rrt.collides(new_start):
                        start_escape_move = (phi, start_node, new_start)
                        start_node = new_start
                        break
                if start_escape_move is None:
                    print('PilotRRTPlanner: Start collides!',e)
                    self.parent.post_event(PilotEvent(StartCollides, args=e.args))
                    self.parent.post_failure()
                    return
                try:
                    (treeA, treeB, path) = self.planner(start_node, goal_node)
                except GoalCollides as e:
                    print('PilotRRTPlanner: Goal collides!',e)
                    self.parent.post_event(PilotEvent(GoalCollides, args=e.args))
                    self.parent.post_failure()
                    return
                except MaxIterations as e:
                    print('PilotRRTPlanner: Max iterations %d exceeded!' % e.args[0])
                    self.parent.post_event(PilotEvent(MaxIterations, args=e.args))
                    self.parent.post_failure()
                    return
                #print('replan',path)

            except GoalCollides as e:
                print('PilotRRTPlanner: Goal collides!',e)
                self.parent.post_event(PilotEvent(GoalCollides, args=e.args))
                self.parent.post_failure()
                return
            except MaxIterations as e:
                print('PilotRRTPlanner: Max iterations %d exceeded!' % e.args[0])
                self.parent.post_event(PilotEvent(MaxIterations, args=e.args))
                self.parent.post_failure()
                return

            if self.parent.verbose:
                print('Path planner generated',len(treeA)+len(treeB),'nodes.')
            if self.parent.robot.path_viewer:
                self.parent.robot.path_viewer.add_tree(path, (1,0,0,0.75))

            self.robot.rrt.draw_path = path

            # Construct the nav plan
            if self.parent.verbose:
                [print(' ',x) for x in path]

            doors = [] # self.robot.world_map.generate_doorway_list()
            navplan = PathPlanner.from_path(path, doors)
            print('navplan=',navplan, '   steps=',navplan.steps)

            # Insert the StartCollides escape move if there is one
            if start_escape_move:
                phi, start, new_start = start_escape_move
                if phi == pi:
                    escape_step = NavStep(NavStep.BACKUP, [new_start])
                    navplan.steps.insert(0, escape_step)
                elif navplan.steps[0].type == NavStep.DRIVE:
                    # Insert at the beginning the original start node we replaced with new_start
                    navplan.steps[0].param.insert(0, start_node)
                else:
                    # Shouldn't get here, but just in case
                    escape_step = NavStep(NavStep.DRIVE, (RRTNode(start.x,start.y), RRTNode(new_start.x,new_start.y)))
                    navplan.steps.insert(0, escape_step)

            #print('finalnavplan steps:', navplan.steps)

            # If no doorpass, we're good to go
            last_step = navplan.steps[-1]
            grid_display = None
            if last_step.type != NavStep.DOORPASS:
                self.post_data((navplan,grid_display))
                return

            # We planned for a doorpass as the last step; replan to the outer gate.
            door = last_step.param
            last_node = navplan.steps[-2].param[-1]
            gate = DoorPass.calculate_gate((last_node.x, last_node.y), door, DoorPass.OUTER_GATE_DISTANCE)
            goal_node = RRTNode(x=gate[0], y=gate[1], q=gate[2])
            print('new goal is', goal_node)
            try:
                (_, _, path) = self.planner(start_node, goal_node)
            except Exception as e:
                print('Pilot replanning for door gateway failed!', e.args)
            cpath = [(node.x,node.y) for node in path]
            navplan = PathPlanner.from_path(cpath, [])
            navplan.steps.append(last_step)  # Add the doorpass step
            self.post_data((navplan,grid_display))

        # ----- End of PilotRRTPlanner -----

    class CheckArrival(StateNode):
        def start(self, event=None):
            super().start(event)
            pose_diff = self.robot.pose - self.parent.target_pose
            distance = (pose_diff.x**2 + pose_diff.y**2) ** 0.5
            MAX_TARGET_DISTANCE = 50.0 # mm
            if distance <= MAX_TARGET_DISTANCE:
                self.post_success()
            else:
                self.post_failure()


    def setup(self):
        #         # PilotToPose machine
        # 
        #         launch: self.ClearDisplays() =N=> planner
        # 
        #         planner: self.PilotRRTPlanner() =D=> recv
        #         planner =PILOT=> ParentPilotEvent() =N=> Print('Path planner failed')
        # 
        #         recv: self.ReceivePlan() =D=> exec
        # 
        #         exec: self.PilotExecutePlan()
        #         exec =C=> check
        #         exec =F=> ParentFails()
        # 
        #         check: self.CheckArrival()
        #         check =S=> ParentCompletes()
        #         check =F=> planner
        
        # Code generated by genfsm on Fri Feb 21 11:00:40 2025:
        
        launch = self.ClearDisplays() .set_name("launch") .set_parent(self)
        planner = self.PilotRRTPlanner() .set_name("planner") .set_parent(self)
        parentpilotevent2 = ParentPilotEvent() .set_name("parentpilotevent2") .set_parent(self)
        print2 = Print('Path planner failed') .set_name("print2") .set_parent(self)
        recv = self.ReceivePlan() .set_name("recv") .set_parent(self)
        exec = self.PilotExecutePlan() .set_name("exec") .set_parent(self)
        parentfails5 = ParentFails() .set_name("parentfails5") .set_parent(self)
        check = self.CheckArrival() .set_name("check") .set_parent(self)
        parentcompletes3 = ParentCompletes() .set_name("parentcompletes3") .set_parent(self)
        
        nulltrans3 = NullTrans() .set_name("nulltrans3")
        nulltrans3 .add_sources(launch) .add_destinations(planner)
        
        datatrans7 = DataTrans() .set_name("datatrans7")
        datatrans7 .add_sources(planner) .add_destinations(recv)
        
        pilottrans2 = PilotTrans() .set_name("pilottrans2")
        pilottrans2 .add_sources(planner) .add_destinations(parentpilotevent2)
        
        nulltrans4 = NullTrans() .set_name("nulltrans4")
        nulltrans4 .add_sources(parentpilotevent2) .add_destinations(print2)
        
        datatrans8 = DataTrans() .set_name("datatrans8")
        datatrans8 .add_sources(recv) .add_destinations(exec)
        
        completiontrans6 = CompletionTrans() .set_name("completiontrans6")
        completiontrans6 .add_sources(exec) .add_destinations(check)
        
        failuretrans6 = FailureTrans() .set_name("failuretrans6")
        failuretrans6 .add_sources(exec) .add_destinations(parentfails5)
        
        successtrans3 = SuccessTrans() .set_name("successtrans3")
        successtrans3 .add_sources(check) .add_destinations(parentcompletes3)
        
        failuretrans7 = FailureTrans() .set_name("failuretrans7")
        failuretrans7 .add_sources(check) .add_destinations(planner)
        
        return self


class PilotPushToPose(PilotToPose):
    def __init__(self,pose):
        super().__init__(pose)
        self.max_turn = 20*(pi/180)

    def planner(self,start_node,goal_node):
        self.robot.rrt.step_size=20
        return self.robot.rrt.plan_push_chip(start_node,goal_node)


class PilotFrustration(StateNode):

    def __init__(self, text_template=None):
        super().__init__()
        self.text_template = text_template  # contains at most one '%s'

    class SayObject(Say):
        def start(self, event=None):
            text_template = self.parent.text_template
            try:
                object_name = self.parent.parent.object.name   # for rooms
            except:
                try:
                    object_name = self.parent.parent.object.id   # for cubes
                except:
                    object_name = None
            if text_template is not None:
                if '%' in text_template:
                    self.text = text_template % object_name
                else:
                    self.text = text_template
            elif object_name is not None:
                self.text = 'Can\'t reach %s' % object_name
            else:
                self.text = 'stuck'
            self.robot.rrt.text = self.text
            super().start(event)


    def setup(self):
        #         launcher: AbortAllActions() =N=> StopAllMotors() =N=> {speak, turn}
        # 
        #         speak: self.SayObject()
        # 
        #         turn: StateNode() =RND=> {left, right}
        # 
        #         left: Turn(5) =C=> left2: Turn(-5)
        # 
        #         right: Turn(-5) =C=> right2: Turn(5)
        # 
        #         {speak, left2, right2} =C(2)=> done
        # 
        #         done: ParentCompletes()
        
        # Code generated by genfsm on Fri Feb 21 11:00:40 2025:
        
        launcher = AbortAllActions() .set_name("launcher") .set_parent(self)
        stopallmotors1 = StopAllMotors() .set_name("stopallmotors1") .set_parent(self)
        speak = self.SayObject() .set_name("speak") .set_parent(self)
        turn = StateNode() .set_name("turn") .set_parent(self)
        left = Turn(5) .set_name("left") .set_parent(self)
        left2 = Turn(-5) .set_name("left2") .set_parent(self)
        right = Turn(-5) .set_name("right") .set_parent(self)
        right2 = Turn(5) .set_name("right2") .set_parent(self)
        done = ParentCompletes() .set_name("done") .set_parent(self)
        
        nulltrans5 = NullTrans() .set_name("nulltrans5")
        nulltrans5 .add_sources(launcher) .add_destinations(stopallmotors1)
        
        nulltrans6 = NullTrans() .set_name("nulltrans6")
        nulltrans6 .add_sources(stopallmotors1) .add_destinations(speak,turn)
        
        randomtrans1 = RandomTrans() .set_name("randomtrans1")
        randomtrans1 .add_sources(turn) .add_destinations(left,right)
        
        completiontrans7 = CompletionTrans() .set_name("completiontrans7")
        completiontrans7 .add_sources(left) .add_destinations(left2)
        
        completiontrans8 = CompletionTrans() .set_name("completiontrans8")
        completiontrans8 .add_sources(right) .add_destinations(right2)
        
        completiontrans9 = CompletionTrans(2) .set_name("completiontrans9")
        completiontrans9 .add_sources(speak,left2,right2) .add_destinations(done)
        
        return self



