
import websocket
import json
import time
import threading
from . import vex
import sys
from typing import Union, cast
import math
import atexit
import signal
import _thread
import pathlib
#module-specific "constant" globals
VERSION_MAJOR = 1
VERSION_MINOR = 0
VERSION_BUILD = 0
VERSION_BETA  = 5
SYS_FLAGS_SOUND_PLAYING     = (1<<0)
SYS_FLAGS_IS_SOUND_DNL      = (1<<16)
SYS_FLAGS_IS_MOVE_ACTIVE    = (1<<1) #if a move_at or move_for (or any "move" command) is active
SYS_FLAGS_IMU_CAL           = (1<<3)
SYS_FLAGS_IS_TURN_ACTIVE    = (1<<4)
SYS_FLAGS_IS_MOVING         = (1<<5) # if there is any wheel movement whatsoever
SYS_FLAGS_HAS_CRASHED       = (1<<6)
SYS_FLAGS_IS_SHAKE          = (1<<8)
SOUND_SIZE_MAX_BYTES     = 255 * 1024
BARREL_MIN_Y             = 160 # *** changed from 180
BARREL_MIN_CX            = 120
BARREL_MAX_CX            = 200

AIVISION_MAX_OBJECTS               = 24
AIVISION_DEFAULT_SNAPSHOT_OBJECTS  =  8
class aim_exception(Exception):
    """VEX AIM Exception Class"""
    pass

class disconnected_exception(aim_exception):
    """Exception that is thrown when robot is not connected over Websocket"""
    pass
class no_image_exception(aim_exception):
    """No image was received"""
    pass

class receive_error_exception(aim_exception):
    """(internally used) error receiving WS frame"""
    pass

class invalid_sound_file_exception(aim_exception):
    """Sound file extension or format is not supported"""

class ws_thread(threading.Thread):
    def __init__ (self, host, ws_name):
        threading.Thread.__init__(self)
        self.host = host
        self.ws_name = ws_name
        self.uri = "ws://%s/%s" %(self.host, self.ws_name)
        self.ws = self.connect_websocket(timeout=4)
        self.callback = None
        self.running = True
        self._ws_needs_reset = False #set equal to true of connection needs to be reset (disconnect, reconnect)

    def connect_websocket(self, timeout):
        ws = websocket.WebSocket()
        try:
            ws.connect(self.uri, timeout=timeout)
        except Exception as error:
            print("Could not connect to %s (reason: %s).  Verify that \"%s\" is the correct IP/hostname of the AIM robot and that it is connected to the same network (AP mode is 192.168.4.1)" %(self.uri, error, self.host))
            sys.exit(1)
        return ws

    def ws_send(self, payload: Union[bytes, str], opcode: int = websocket.ABNF.OPCODE_TEXT):
        try:
            self.ws.send(payload, opcode)
        except (ConnectionResetError, websocket.WebSocketException) as error:
            self._ws_needs_reset = True
            raise disconnected_exception("%s: error sending data to robot, apparently disconnected, will try to reconnect; error: '%s'" %(self.ws_name, error))

    def ws_receive(self):
        try:
            data = self.ws.recv()
        except (ConnectionResetError, websocket.WebSocketException) as error:
            self._ws_needs_reset = True
            raise receive_error_exception("%s: error receiving data from robot, apparently disconnected, will try to reconnect; error: '%s'" %(self.ws_name, error))
        return data
    
    def ws_close(self):
        try:
            self.ws.close()
        except Exception as error:
            print("couldn't close Websocket connection, already closed? error: %s" %error)

class ws_status_thread(ws_thread):
    def __init__(self, host):
        super().__init__ (host, "ws_status")
        self._empty_status = {
            "controller": {"flags": "0x0000", "stick_x": 0, "stick_y": 0, "battery": 0},
            "robot": {
                "flags": "0x00000000",
                "battery": 0,
                "touch_flags": "0x0000",
                "touch_x": 0,
                "touch_y": 0,
                "robot_x": 0,
                "robot_y": 0,
                "roll": "0",
                "pitch": "0",
                "yaw": "0",
                "heading": "0",
                "acceleration": {"x": "0", "y": "0", "z": "0"},
                "gyro_rate": {"x": "0", "y": "0", "z": "0"},
            },
            "aivision": {
                "classnames": {
                    "count": 4,
                    "items": [
                        {"index": 0, "name": "Ball"},
                        {"index": 1, "name": "BlueBarrel"},
                        {"index": 2, "name": "OrangeBarrel"},
                        {"index": 3, "name": "Robot"},
                    ],
                },
                "objects": {
                    "count": 0,
                    "items": [
                        {
                            "type": 0,
                            "id": 0,
                            "type_str": "0",
                            "originx": 0,
                            "originy": 0,
                            "width": 0,
                            "height": 0,
                            "score": 0,
                            "name": "0",
                        }
                    ]},
            },
        }
        self.current_status = self._empty_status
        self.is_move_active_flag_needs_setting    = False
        self.is_turn_active_flag_needs_setting    = False
        self.is_moving_flag_needs_setting         = False
        self.is_moving_flag_needs_clearing        = False
        self.imu_cal_flag_needs_setting           = False
        self.sound_playing_flag_needs_setting     = False
        self.sound_downloading_flag_needs_setting = False

        self._packets_lost_counter = 0
        self.heartbeat = 0

    def update_status_flags(self):
        if self.is_move_active_flag_needs_setting:
            self.set_is_move_active_flag()
            self.is_move_active_flag_needs_setting = False

        if self.is_turn_active_flag_needs_setting:
            self.set_is_turn_active_flag()
            self.is_turn_active_flag_needs_setting = False

        if self.is_moving_flag_needs_setting:
            self.set_is_moving_flag()
            self.is_moving_flag_needs_setting = False

        if self.is_moving_flag_needs_clearing:
            self.clear_is_moving_flag()
            self.is_moving_flag_needs_clearing = False

        if self.imu_cal_flag_needs_setting:
            self.set_imu_cal_flag()
            self.imu_cal_flag_needs_setting = False

        if self.sound_playing_flag_needs_setting:
            self.set_sound_playing_flag()
            self.sound_playing_flag_needs_setting = False

        if self.sound_downloading_flag_needs_setting:
            self.set_sound_downloading_flag()
            self.sound_downloading_flag_needs_setting = False

    def set_is_move_active_flag(self):
        robot_flags = self.current_status["robot"]["flags"]
        #update is_move_active flag (convert robot_flags to int, set bit 1 to 1, then convert back to hex string)
        new_robot_flags = hex(int(robot_flags, 16) | SYS_FLAGS_IS_MOVE_ACTIVE)
        self.current_status["robot"]["flags"] = new_robot_flags

    def set_is_turn_active_flag(self):
        robot_flags = self.current_status["robot"]["flags"]
        #update is_turn_active flag (convert robot_flags to int, set bit 1 to 1, then convert back to hex string)
        new_robot_flags = hex(int(robot_flags, 16) | SYS_FLAGS_IS_TURN_ACTIVE)
        self.current_status["robot"]["flags"] = new_robot_flags

    def set_is_moving_flag(self):
        robot_flags = self.current_status["robot"]["flags"]
        #update is_moving flag (convert robot_flags to int, set bit 1 to 1, then convert back to hex string)
        new_robot_flags = hex(int(robot_flags, 16) | SYS_FLAGS_IS_MOVING)
        self.current_status["robot"]["flags"] = new_robot_flags
    
    def clear_is_moving_flag(self):
        robot_flags = self.current_status["robot"]["flags"]
        #update is_moving flag (convert robot_flags to int, set bit 1 to 1, then convert back to hex string)
        new_robot_flags = hex(int(robot_flags, 16) & ~SYS_FLAGS_IS_MOVING)
        self.current_status["robot"]["flags"] = new_robot_flags

    def set_imu_cal_flag(self):
        robot_flags = self.current_status["robot"]["flags"]
        #update imu_cal flag (convert robot_flags to int, set bit 1 to 1, then convert back to hex string)
        new_robot_flags = hex(int(robot_flags, 16) | SYS_FLAGS_IMU_CAL)
        self.current_status["robot"]["flags"] = new_robot_flags

    def set_sound_playing_flag(self):
        robot_flags = self.current_status["robot"]["flags"]
        #update sound_playing flag (convert robot_flags to int, set bit 1 to 1, then convert back to hex string)
        new_robot_flags = hex(int(robot_flags, 16) | SYS_FLAGS_SOUND_PLAYING)
        self.current_status["robot"]["flags"] = new_robot_flags

    def set_sound_downloading_flag(self):
        robot_flags = self.current_status["robot"]["flags"]
        #update is_sound_downloading flag (convert robot_flags to int, set bit 1 to 1, then convert back to hex string)
        new_robot_flags = hex(int(robot_flags, 16) | SYS_FLAGS_IS_SOUND_DNL)
        self.current_status["robot"]["flags"] = new_robot_flags

    def check_shake_flag(self):
        robot_flags = self.current_status["robot"]["flags"]
        if int(robot_flags, 16) & SYS_FLAGS_IS_SHAKE != 0:
            _thread.interrupt_main()

    def check_crash_flag(self):
        robot_flags = self.current_status["robot"]["flags"]
        if int(robot_flags, 16) & SYS_FLAGS_HAS_CRASHED != 0:
            print('has crashed')

    def is_current_status_empty(self):
        if self.current_status == self._empty_status:
            return True
        else:
            return False

    def run(self):
        while self.running:
            new_status_JSON = ''
            new_status = self._empty_status
            if self._ws_needs_reset:
                self.ws_close()
                self._ws_needs_reset = False

            if self.ws.connected:
                status_packet_error = False
                self.ws_send((1).to_bytes(1, 'little'), websocket.ABNF.OPCODE_BINARY)
                try:
                    new_status_JSON = self.ws_receive()
                except:
                    status_packet_error = True

                if not status_packet_error:
                    try:
                        new_status = json.loads(new_status_JSON)
                    except:
                        status_packet_error = True

                # If we have an error receiving packet, initially we want to keep current_status unchanged.  After enough dropped packets, set current_status to empty values.
                if status_packet_error:
                    self._packets_lost_counter += 1
                    print("lost a status packet, counter: %d" %self._packets_lost_counter)
                #we have received a valid status packet, so update current_status:
                else:
                    self._packets_lost_counter = 0  #reset this counter
                    self.current_status = new_status
                    # print("current_status: ", self.current_status)
                    if self.callback:
                        self.callback()

                    # if a certain commands are sent, the very next status flag won't have the appropriate flags set yet, so over-ride locally.
                    self.update_status_flags()
                    self.check_shake_flag()
                    self.check_crash_flag()
                    self.heartbeat = not self.heartbeat

                if self._packets_lost_counter > 5:
                    self.current_status = self._empty_status

                # print("current_status: ", self.current_status)
                time.sleep(0.05)
            else:
                # print ("trying to reconnect to ws_status")
                try:
                    print("%s reconnecting" %self.ws_name)
                    self.ws.connect(self.uri)
                except:
                    pass # we'll keep trying to reconnect
class ws_img_thread(ws_thread):
    def __init__(self, host):
        super().__init__ (host, "ws_img")
        self.current_image_index = 0
        self.image_list: list[bytes] = [bytes(1), bytes(1)]
    
        self._next_image_index = 1
        self._streaming = False

    def run(self):
        while self.running:
            if self._ws_needs_reset:
                self.ws_close()
                self._ws_needs_reset = False

            if self.ws.connected and self._streaming:
                if self.current_image_index == 1:
                    self._next_image_index    = 1
                    self.current_image_index = 0
                else:
                    self._next_image_index    = 0
                    self.current_image_index = 1

                try:
                    self.image_list[self._next_image_index] = cast(bytes, self.ws_receive()) # cast to narrow str | bytes down to bytes
                    if self.callback:
                        self.callback()
                except receive_error_exception:
                    self.image_list[self._next_image_index] = (0).to_bytes(1, 'little')

            elif self.ws.connected == False:
                self._streaming = False
                try:
                    print("%s reconnecting" %self.ws_name)
                    self.ws.connect(self.uri)
                except:
                    pass # we'll keep trying to reconnect
            else:
                time.sleep(0.05)
    def stop_stream(self):
        self._streaming = False
        self.ws_send((0).to_bytes(1, 'little'), websocket.ABNF.OPCODE_BINARY)

    def start_stream(self):
        self._streaming = True
        self.ws_send((1).to_bytes(1, 'little'), websocket.ABNF.OPCODE_BINARY)

class ws_cmd_thread(ws_thread):
    def __init__(self, host):
        super().__init__ (host, "ws_cmd")

    def run(self):
        while self.running:
            if self._ws_needs_reset:
                self.ws_close()
                self._ws_needs_reset = False

            if self.ws.connected == False:
                try:
                    print("%s reconnecting" %self.ws_name)
                    self.ws.connect(self.uri)
                except:
                    pass # we'll keep on trying

            time.sleep(0.2)

class ws_audio_thread(ws_thread):
    def __init__(self, host):
        super().__init__ (host, "ws_audio")

    def run(self):
        while self.running:
            if self._ws_needs_reset:
                self.ws_close()
                self._ws_needs_reset = False

            if self.ws.connected == False:
                try:
                    print("%s reconnecting" %self.ws_name)
                    self.ws.connect(self.uri)
                except:
                    pass # we'll keep on trying

            time.sleep(0.2)

class Robot():
    """AIM Robot class.  When initializing, provide a host (IP address, hostname, or even domain name) or leave at empty for default if AIM is in WiFi AP mode."""

    def __init__(self, host="192.168.4.1"):
        """
        Initialize the Robot with default settings and WebSocket connections.
        """
        print("Welcome to the AIM Websocket Python Client. Running version %d.%d.%d.%d and connecting to %s" %(VERSION_MAJOR, VERSION_MINOR, VERSION_BUILD, VERSION_BETA, host))
        self.move_active_cmd_list = ["drive", "drive_for"]
        self.turn_active_cmd_list = [ "turn", "turn_for", "turn_to"]
        self.stopped_active_cmd_list = self.move_active_cmd_list + self.turn_active_cmd_list
        self.host = host

        self.inertial = Gyro(self)
        self.aiv      = AiVision(self)
        self.screen   = Screen(self)
        self._ws_status_thread        = ws_status_thread(self.host)
        self._ws_status_thread.daemon = True
        self._ws_status_thread.start()

        self._ws_img_thread           = ws_img_thread(self.host)
        self._ws_img_thread.daemon    = True
        self._ws_img_thread.start()

        self._ws_cmd_thread           = ws_cmd_thread(self.host)
        self._ws_cmd_thread.daemon    = True
        self._ws_cmd_thread.start()

        self._ws_audio_thread         = ws_audio_thread(self.host)
        self._ws_audio_thread.daemon  = True
        self._ws_audio_thread.start()

        atexit.register(self.exit_handler)
        #signal.signal(signal.SIGINT, self.kill_handler)
        signal.signal(signal.SIGTERM, self.kill_handler)

        self._program_init()

        self.drive_speed = 75
        self.turn_speed  = 50

        # We don't want to execute certain things (like reset_heading) until we start getting status packets
        while self._ws_status_thread.is_current_status_empty() == True:
            # print("waiting for status")
            time.sleep(0.05)
        self.inertial.reset_heading()

    def exit_handler(self):
        """upon system exit, either due to SIGINT/SIGTERM or uncaught exceptions"""
        if hasattr(self, '_ws_cmd_thread'): #if connection were never established, this property wouldn't exist
            print("program terminating, stopping robot")
            try:
                self.stop_all_movement()
            except Exception as error:
                print("exceptions arose during stop_all_movement(), error:", error)
        else:
            print("program terminating (never connected to robot)")

        try:
            self._ws_cmd_thread.running = False
        except:
            # print("ws_cmd_thread doesn't exist")
            pass
        try:
            self._ws_status_thread.running = False
        except:
            # print("ws_status_thread doesn't exist")
            pass
        try:
            self._ws_img_thread.running    = False
        except:
            # print("ws_img_thread doesn't exist")
            pass
        try:
            self._ws_img_thread.stop_stream()
        except:
            # print("error stopping ws_img stream")
            pass

    def kill_handler(self, signum, frame):
        """when kill signal is received, exit the program.  Will result in exit_handler being run"""
        signame = signal.Signals(signum).name
        print(f'Received signal {signame} ({signum})')
        sys.exit(0)

    def __getattribute__(self, name):
        """This function gets called before any other Robot function.\n
        If we are not connected to the robot (just looking at ws_cmd), then raise an exception
        and terminate program unless the user handles the exception."""
        method = object.__getattribute__(self, name)
        if not method:
            # raise Exception("Method %s not implemented" %name)
            return
        if callable(method):
            if self._ws_cmd_thread.ws.connected == False:
                raise disconnected_exception("error calling %s: not connected to robot" %name)
        return method
       
    @property
    def status(self):
        return self._ws_status_thread.current_status

    def robot_send(self, json_cmd):
        disconnected_error = False
        json_cmd_string = json.dumps(json_cmd, separators=(',',':'))
        # print("sending: ", json_cmd_string)
        try:
            cmd_id = json_cmd["cmd_id"]
        except:
            print("robot_send did not receive a cmd_id")
            return

        self._ws_cmd_thread.ws_send(str.encode(json_cmd_string), websocket.ABNF.OPCODE_BINARY)
        try:
            response_JSON = self._ws_cmd_thread.ws_receive()
        except receive_error_exception:
            disconnected_error = True
            raise disconnected_exception("robot got disconnected after sending cmd_id: %s" %cmd_id) from None # disable exception chaining
            # not trying to resend command because that would take too long, let user decide.
  
        try:
            response = json.loads(response_JSON)
        except Exception as error:
            print("%s Error: could not parse ws_cmd JSON response: '%s'" %(cmd_id, error))
            print("response_JSON", response_JSON)
            return

        # print("response_JSON", response_JSON)
        if response["cmd_id"] == "cmd_unknown":
            print("robot: did not recognize command: ", cmd_id)
            return

        if response["status"] == "error":
            try:
                error_info_string = response["error_info"]
            except KeyError:
                error_info_string = "no reason given"
            print("robot: error processing command, reason: ", error_info_string)
            return

        # trigger a local update to the robot status flags in ws_status_thread
        if response["status"] in ["complete", "in_progress"]:
            if response["cmd_id"] in self.move_active_cmd_list:
                self._ws_status_thread.is_move_active_flag_needs_setting = True
            if response["cmd_id"] in self.turn_active_cmd_list:
                self._ws_status_thread.is_turn_active_flag_needs_setting = True
            if response["cmd_id"] in self.stopped_active_cmd_list:
                self._ws_status_thread.is_moving_flag_needs_setting  = True
                self._ws_status_thread.is_moving_flag_needs_clearing = False
            if response["cmd_id"] == "imu_calibrate":
                self._ws_status_thread.imu_cal_flag_needs_setting = True

        return

    def robot_send_audio(self, audio):
        self._ws_audio_thread.ws_send(audio, websocket.ABNF.OPCODE_BINARY)

    def _program_init(self):
        """Sends a command indicating to robot that new program is starting.  To be called during __init__"""
        json = { "cmd_id" : "program_init" }
        self.robot_send(json)

    def get_battery_capacity(self):
        """Get the remaining capacity of the battery (relative state of charge) in percent."""
        battery_capacity = self.status["robot"]["battery"]
        return battery_capacity

    def get_roll(self):
        value = self.status["robot"]["roll"]
        if type(value) == str:
            value = float(value)
        return value

    def get_pitch(self):
        value = self.status["robot"]["pitch"]
        if type(value) == str:
            value = float(value)
        return value
    
    def get_yaw(self):
        value = self.status["robot"]["yaw"]
        if type(value) == str:
            value = float(value)
        return value

    def get_heading(self):
        raw_heading = self.status["robot"]["heading"]
        if type(raw_heading) == str:
            raw_heading = float(raw_heading)
        heading = math.fmod(raw_heading - self.inertial.heading_offset, 360)
        if heading < 0:
            heading += 360
        return heading

    def get_heading_raw(self):
        raw_heading = self.status["robot"]["heading"]
        if type(raw_heading) == str:
            raw_heading = float(raw_heading)
        return raw_heading

    def get_x(self):
        origin_x = float(self.status["robot"]["robot_x"])
        origin_y = float(self.status["robot"]["robot_y"])
        # print("raw:", origin_x, origin_y)
        offset_radians = -math.radians(self.inertial.heading_offset)
        x = origin_x * math.cos(offset_radians) + origin_y * math.sin(offset_radians)

        return x

    def get_y(self):
        origin_x = float(self.status["robot"]["robot_x"])
        origin_y = float(self.status["robot"]["robot_y"])
        offset_radians = -math.radians(self.inertial.heading_offset)
        y = origin_y * math.cos(offset_radians) - origin_x * math.sin(offset_radians)

        return y

    def is_move_active(self):
        """returns true if a move_at() or move_for() command is active with nonzero speed"""
        if self._ws_status_thread.is_move_active_flag_needs_setting:
            return True
        robot_flags = self.status["robot"]["flags"]
        is_move_active = bool(int(robot_flags, 16) & SYS_FLAGS_IS_MOVE_ACTIVE)
        return is_move_active

    def is_turn_active(self):
        """returns true if a turn(), turn_to(), or turn_for() command is active with nonzero speed"""
        if self._ws_status_thread.is_turn_active_flag_needs_setting:
            return True
        robot_flags = self.status["robot"]["flags"]
        is_turn_active = bool(int(robot_flags, 16) & SYS_FLAGS_IS_TURN_ACTIVE)
        return is_turn_active
    
    def is_stopped(self):
        """returns true if no move, turn, or spin_wheels command is active (i.e. no wheels should be moving)"""
        if self._ws_status_thread.is_moving_flag_needs_clearing:
            return True
        if self._ws_status_thread.is_moving_flag_needs_setting:
            return False
        robot_flags = self.status["robot"]["flags"]
        is_stopped = not bool(int(robot_flags, 16) & SYS_FLAGS_IS_MOVING)
        return is_stopped
    # Drive methods
    def move_at(self, angle, drive_speed=None):
        if drive_speed == None:
            drive_speed = self.drive_speed
        json = {"cmd_id": "drive", "angle": angle, "speed": drive_speed}
        self.robot_send(json)

    def move_for(self, distance, angle, drive_speed=None, turn_speed=None, wait=True):
        if drive_speed == None:
            drive_speed = self.drive_speed
        if turn_speed == None:
            turn_speed = self.turn_speed
        # if not final_heading:
        #     final_heading = self.get_heading_raw()

        json = {
            "cmd_id"        : "drive_for",
            "distance"      : distance,
            "angle"         : angle, #drive angle
            # "final_heading" : final_heading,
            "final_heading" : 0, # final_heading not implemented yet
            "drive_speed"   : drive_speed,
            "turn_speed"    : turn_speed,
        }
        self.robot_send(json)
        if wait:
            self._block_on_state(self.is_move_active)

    def turn(self, turn_direction: vex.TurnType, turn_speed=None):
        """turn indefinitely at turn_rate"""
        if turn_speed == None:
            turn_speed = self.turn_speed
        if turn_direction == vex.TurnType.LEFT:
            turn_speed = -turn_speed
        json = {"cmd_id": "turn", "turn_rate": turn_speed}
        self.robot_send(json)
    
    def stop_all_movement(self):
        self.move_at(0,0)
        self.turn(vex.TurnType.RIGHT, 0)
        self._ws_status_thread.clear_is_moving_flag() # clear the is_moving_flag now (used by robot.is_stopped())
        # trigger this flag to be cleared again next time a status msg is received, in case the robot hasn't update the state yet:
        self._ws_status_thread.is_moving_flag_needs_clearing = True 
        self._ws_status_thread.is_moving_flag_needs_setting  = False

    def _block_on_state(self, state_method):
        time_start = time.time()
        blocking = True
        while True:
            if state_method() == False: # debounce
                time.sleep(0.05)
                if state_method() == False:
                    break
            # print("blocking")
            time_elapsed = time.time() - time_start
            time.sleep(0.1)
            #if turning/moving took too long, we want to stop moving and stop blocking.
            if time_elapsed > 10:
                print("%s wait timed out, stopping" %state_method.__name__)
                self.stop_all_movement()
                return

    def turn_to(self, heading, turn_speed=None, wait=True):
        """turn to a heading (degrees) at turn_rate (deg/sec)\n
        heading can be -360 to 360"""
        if not (-360 < heading < 360):
            raise ValueError("heading must be between -360 and 360")
        if turn_speed == None:
            turn_speed = self.turn_speed        
        heading_offset = self.inertial.heading_offset
        heading = math.fmod (heading_offset + heading, 360)
        json = {"cmd_id": "turn_to", "heading": heading, "turn_rate": turn_speed}
        self.robot_send(json)
        if wait:
            self._block_on_state(self.is_turn_active)

    def turn_for(self, turn_direction: vex.TurnType, angle, turn_speed=None, wait=True):
        """turn for a 'angle' number of degrees at turn_rate (deg/sec)"""
        if turn_speed == None:
            turn_speed = self.turn_speed        
        if turn_direction == vex.TurnType.LEFT:
            angle = -angle
        json = {"cmd_id": "turn_for", "angle": angle, "turn_rate": turn_speed}
        self.robot_send(json)
        if wait:
            self._block_on_state(self.is_turn_active)

    def set_move_velocity(self, velocity):
        self.drive_speed = velocity

    def set_turn_velocity(self, velocity):
        self.turn_speed = velocity

    def spin_wheels(self, velocity1: int, velocity2: int, velocity3: int):
        json = {"cmd_id": "spin_wheels", "vel1": velocity1, "vel2": velocity2, "vel3": velocity3}
        self.robot_send(json)

    def set_pose(self, x, y, heading):
        """set x, y, and heading to the input values"""
        self.inertial.set_heading(heading)
        offset_radians = -math.radians(self.inertial.heading_offset)
        origin_x = x * math.cos(offset_radians) - y * math.sin(offset_radians)        
        origin_y = y * math.cos(offset_radians) + x * math.sin(offset_radians)
        json = {"cmd_id": "set_pose", "x" : origin_x, "y" : origin_y}
        self.robot_send(json)
        for status_counter in range(2):
            heartbeat_state = self._ws_status_thread.heartbeat
            while heartbeat_state == self._ws_status_thread.heartbeat: # wait till we get a new status packet
                # print("status_counter %d" %(status_counter))
                time.sleep(0.050)

    def set_light_color(self, *args):
        """Sets the color of any one of six LEDs, with RGB values \n
        Each call to this function results in a Websocket command. \n
        Example: robot.set_light_color(vex.LightType.ALL, vex.Color.BLUE)"""
        if len(args) >= 1 and type(args) != vex.LightType:
                light_index = args[0]
        else:
            raise TypeError("first argument must be of type", vex.LightType)

        if len(args) == 2 and isinstance(args[1], (vex.Color, vex.Color.DefinedColor)):
            r = (args[1].value >> 16) & 0xFF
            g = (args[1].value >>  8) & 0xFF
            b =  args[1].value & 0xFF

        elif len(args) == 4:
            r = args[1]
            g = args[2]
            b = args[3]

        else:
            raise TypeError("bad parameters, n_args: %d" %(len(args)))
        json = {"cmd_id": "light_set", light_index: {"r": r, "g": g, "b": b}}
        self.robot_send(json)

    def set_heading(self, heading):
        """robot heading will be set to `heading`"""
        self.inertial.set_heading(heading)
    # def kick(self, strength):
    #     json = {"cmd_id": "kick", "strength": strength}
    #     return self.robot_send(json)
    
    def kick(self, kick_type: vex.KickType):
        """kick at specified kick strength"""
        json = {"cmd_id": kick_type}
        self.robot_send(json)

    def push(self):
        """set object down.  Same as kick(vex.KickType.SOFT) for now)"""
        # json = {"cmd_id": "push"}
        self.kick(vex.KickType.SOFT)

    def play_sound(self, sound_type: vex.SoundType, volume):
        """Play a sound from a list of preset sounds"""
        json = {"cmd_id": "play_sound", "name": sound_type, "volume": volume}
        self.robot_send(json)
   
    def play_sound_file(self, filepath: str, volume = 100):
        """play a WAV or MP3 file stored on the client side; file will be transmitted to robot\n
           Maximum filesize is 255 KB"""
        file = pathlib.Path(filepath)
        size = file.stat().st_size
        if size > SOUND_SIZE_MAX_BYTES:
            raise invalid_sound_file_exception("file size of %d bytes is too big; max size allowed is %d bytes (%.1f kB)" %(size, SOUND_SIZE_MAX_BYTES, SOUND_SIZE_MAX_BYTES/1024))
        
        extension = file.suffix
        filename = file.name
        audio = bytearray(64)
        if not (extension == ".wav" or extension == ".mp3"):
            raise invalid_sound_file_exception("extension is %s; expected extension to be wav or mp3" %extension)
        try:
            f = open(filepath, 'rb')
        except FileNotFoundError:
            print ("File", filepath, "was not found")
        else:
            with f:
                data = f.read()

            # do some sanity checks to make sure it's really a wave:
            if extension == ".wav":
                if not (data[0:4] == b'RIFF' and data[8:12] == b'WAVE'):
                    raise invalid_sound_file_exception("file extension was .wav but does not appear to actually be a WAVE file")
                channels = int.from_bytes(data[22:24], "little")
                if channels > 2:
                    raise invalid_sound_file_exception("only mono or stereo is supported, detected %d channels." %channels)
                if channels == 2:
                    print("%s is stereo; mono is recommended")
                # first 64 bytes of audio is header
                audio[0:1] = (0).to_bytes(1, 'little')

            # assuming the mp3 is valid:
            elif extension == ".mp3":
                audio[0:1] = (1).to_bytes(1, 'little') 

            # set volume
            audio[1:2] = (volume).to_bytes(1, 'little') 

            audio[4:8] = (len(data)).to_bytes(4, 'little') # length of data
            audio[8:12] = (0).to_bytes(4, 'little') # file chunk number
            audio[32:32+len(filename)] = map(ord, filename[:32]) # filename
            audio.extend(data) # append the data
            self.robot_send_audio(audio)
            self._ws_status_thread.set_sound_playing_flag()
            self._ws_status_thread.set_sound_downloading_flag()
            self._ws_status_thread.sound_downloading_flag_needs_setting = True # have it be set again after next status message
            self._ws_status_thread.sound_playing_flag_needs_setting = True # have it be set again after next status message

    def is_sound_active(self):
        """returns true if sound is currently playing or if it is being transmitted for playing"""
        robot_flags = self.status["robot"]["flags"]
        sound_active = bool(int(robot_flags, 16) & SYS_FLAGS_SOUND_PLAYING) or bool(int(robot_flags, 16) & SYS_FLAGS_IS_SOUND_DNL)
        return sound_active

    def get_camera_image(self):
        """returns a camera image; starts stream when first called; first image will take about 0.3 seconds to return.\n
           Subsequently, images will continually stream from robot and therefore will be immediately available."""
        if self._ws_img_thread._streaming == False:
            # print("starting the stream")
            start_time = time.time()
            time_elapsed = 0
            self._ws_img_thread.start_stream()
            while (self._ws_img_thread.image_list[self._ws_img_thread.current_image_index] == bytes(1) and time_elapsed < 0.5):
                time.sleep(0.01)
                time_elapsed = time.time() - start_time
        image = self._ws_img_thread.image_list[self._ws_img_thread.current_image_index]
        if image == bytes(1):
            raise no_image_exception("no image was received")
        return image

    def take_snapshot(self, type, count=8):
        return self.aiv.take_snapshot(type, count)
    
    def largest_object(self):
        '''### Request the largest object from the last take_snapshot(...) call

        #### Arguments:
            None

        #### Returns:
            An AiVisionObject object or None if it does not exist
        '''        
        return self.aiv.largest_object()
     
    def object_count(self):
        return self.aiv.object_count()
    
    def has_barrel(self):
        return self.aiv.has_barrel()
    
    def has_ball(self):
        return self.aiv.has_ball()

    def show_emoji(self, emoji: vex.Emoji.EmojiType, look: vex.EmojiLook.EmojiLookType = vex.EmojiLook.CENTER):
        return self.screen.show_emoji(emoji, look)

    def hide_emoji(self):
        return self.screen.hide_emoji()
class Gyro():
    def __init__(self, robot_instance: Robot):
        """
        Initialize the Gyro with default settings.
        """
        self.robot_instance = robot_instance
        self.heading_offset = 0

    def calibrate(self):
        """calibrate the IMU.  Can't check if calibration is done, so probably do not call for now"""
        json = {"cmd_id": "imu_calibrate"}
        self.robot_instance.robot_send(json)
    
    def is_calibrating(self):
        if self.robot_instance._ws_status_thread.imu_cal_flag_needs_setting == True:
            return True
        robot_flags = self.robot_instance._ws_status_thread.current_status["robot"]["flags"]
        calibrating = bool(int(robot_flags, 16) & SYS_FLAGS_IMU_CAL)
        return calibrating

    def acceleration(self, axis: vex.AxisType):
        if   axis == vex.AxisType.XAXIS:
            value = self.robot_instance._ws_status_thread.current_status["robot"]["acceleration"]["x"]
        elif axis == vex.AxisType.YAXIS:
            value = self.robot_instance._ws_status_thread.current_status["robot"]["acceleration"]["y"]
        elif axis == vex.AxisType.ZAXIS:
            value = self.robot_instance._ws_status_thread.current_status["robot"]["acceleration"]["z"]
        if type(value) == str:
            value = float(value)
        return value

    def gyro_rate(self, axis: vex.AxisType):
        if   axis == vex.AxisType.XAXIS:
            value = self.robot_instance._ws_status_thread.current_status["robot"]["gyro_rate"]["x"]
        elif axis == vex.AxisType.YAXIS:
            value = self.robot_instance._ws_status_thread.current_status["robot"]["gyro_rate"]["y"]
        elif axis == vex.AxisType.ZAXIS:
            value = self.robot_instance._ws_status_thread.current_status["robot"]["gyro_rate"]["z"]
        if type(value) == str:
            value = float(value)
        return value

    def set_heading(self, heading):
        """robot heading will be set to `heading`"""
        raw_heading = self.robot_instance.get_heading_raw()
        # print("reset heading to %f, get_heading_raw(): %f" %(heading, raw_heading))
        self.heading_offset = raw_heading - heading

    def reset_heading(self):
        """robot heading will be set to 0"""
        self.set_heading(0)

class Colordesc:
    '''### Colordesc class - a class for holding an AI vision sensor color definition

    #### Arguments:
        index : The color description index (1 to 7)
        red : the red color value
        green : the green color value
        blue : the blue color value
        hangle : the range of allowable hue
        hdsat : the range of allowable saturation

    #### Returns:
        An instance of the Colordesc class

    #### Examples:
        COL1 = Colordesc(1,  13, 114, 227, 10.00, 0.20)\\
        COL2 = Colordesc(2, 237,  61,  74, 10.00, 0.20)\\
    '''
    def __init__(self, index, red, green, blue, hangle, hdsat):
        self.id = index
        pass

class Codedesc:
    '''### Codedesc class - a class for holding AI vision sensor codes

    A code description is a collection of up to five AI vision color descriptions.
    #### Arguments:
        index : The code description index (1 to 5)
        c1 : An AI vision Colordesc
        c1 : An AI vision Colordesc
        c3 (optional) : An AI vision Colordesc
        c4 (optional) : An AI vision Colordesc
        c5 (optional) : An AI vision Colordesc

    #### Returns:
        An instance of the Codedesc class

    #### Examples:
        COL1 = Colordesc(1,  13, 114, 227, 10.00, 0.20)\\
        COL2 = Colordesc(2, 237,  61,  74, 10.00, 0.20)\\
        C1 = Codedesc( 1, COL1, COL2 )
    '''
    def __init__(self, index, c1:Colordesc, c2:Colordesc, *args):
        self.id = index
        pass

class Tagdesc:
    '''### Tagdesc class - a class for holding AI vision sensor tag id

    A tag description holds an apriltag id
    #### Arguments:
        id : The apriltag id (positive integer, not 0)

    #### Returns:
        An instance of the Tagdesc class

    #### Examples:
        T1 = Tagdesc( 23 )
    '''
    def __init__(self, index):
        self.id = index
        pass

class AiObjdesc:
    '''### AiObjdesc class - a class for holding AI vision sensor AI object id

    A tag description holds an apriltag id
    #### Arguments:
        id : The AI Object (model) id (positive integer, not 0)

    #### Returns:
        An instance of the AiObjdesc class

    #### Examples:
        A1 = AiObjdesc( 2 )
    '''
    def __init__(self, index):
        self.id = index
        pass

class ObjDesc:
     def __init__(self, index):
        self.id = index
        pass   
class _ObjectTypeMask:
    unkownObject = 0
    colorObject  = (1 << 0)
    codeObject   = (1 << 1)
    modelObject  = (1 << 2)
    tagObject    = (1 << 3)
    allObject    = (0x3F)

MATCH_ALL_ID = 0xFFFF
class AiVision():
    ALL_TAGS = Tagdesc(MATCH_ALL_ID)
    '''A tag description for take_snapshot indicating all tag objects to be returned'''
    ALL_COLORS = Colordesc(MATCH_ALL_ID, 0, 0, 0, 0, 0)
    '''A tag description for take_snapshot indicating all color objects to be returned'''
    ALL_CODES = Codedesc(MATCH_ALL_ID, ALL_COLORS, ALL_COLORS)
    '''A tag description for take_snapshot indicating all code objects to be returned'''
    ALL_AIOBJS = AiObjdesc(MATCH_ALL_ID)
    '''A tag description for take_snapshot indicating all AI model objects to be returned'''
    ALL_OBJECTS = ObjDesc(MATCH_ALL_ID)
    '''A tag description for take_snapshot indicating all objects to be returned'''


    def __init__(self, robot_instance: Robot):
        self.robot_instance     = robot_instance
        self._object_count_val  = 0
        self._largest_object    = None

    def take_snapshot(self, type, count=8):
        '''### Request the AI vision sensor to filter latest objects to match color,\\
               code, apriltag or AI object.

        #### Arguments:
            type : A color, code or other object type
            count (optional) : the maximum number of objects to obtain.  default is 8.

        #### Returns:
            tuple of AiVisionObject, this will be an empty tuple if nothing is available.

        #### Examples:
            # look for and return 1 object matching COL1\\
            objects = aivision1.take_snapshot(COL1)

            # look for and return a maximum of 4 objects matching SIG_1\\
            objects = aivision1.take_snapshot(COL1, 4)

            # return apriltag objects\\
            objects = aivision1.take_snapshot(ALL_TAGS, AIVISION_MAX_OBJECTS)
        '''
        if isinstance(type, Colordesc):
            type_mask = _ObjectTypeMask.colorObject
            id = type.id
        elif isinstance(type, Codedesc):
            type_mask = _ObjectTypeMask.codeObject
            id = type.id
        elif isinstance(type, AiObjdesc):
            type_mask = _ObjectTypeMask.modelObject
            id = type.id
        elif isinstance(type, Tagdesc):
            type_mask = _ObjectTypeMask.tagObject
            id = type.id
        elif isinstance(type, ObjDesc):
            type_mask = _ObjectTypeMask.allObject
            id = type.id
        else:
            type_mask = _ObjectTypeMask.colorObject # default value
            id = type # assume the first argument is really the Colordesc id.

        if count > AIVISION_MAX_OBJECTS:
            count = AIVISION_MAX_OBJECTS
    
        objects = self.robot_instance.status["aivision"]["objects"]
        item_count = objects["count"]
        ai_object_list = [AiVisionObject() for item in range(item_count)]
        # first just extract everything we got from ws_status
        for item in range(item_count):
            ai_object_list[item].type       = objects["items"][item]["type"]
            ai_object_list[item].id         = objects["items"][item]["id"]
            ai_object_list[item].originX    = objects["items"][item]["originx"]
            ai_object_list[item].originY    = objects["items"][item]["originy"]
            ai_object_list[item].width      = objects["items"][item]["width"]
            ai_object_list[item].height     = objects["items"][item]["height"]

            if ai_object_list[item].type ==  _ObjectTypeMask.modelObject: #AI model objects can have a classname
                ai_object_list[item].classname  = self.robot_instance.status["aivision"]["classnames"]["items"][ai_object_list[item].id]["name"]
        # print("diagnostic: ai_object_list: ", ai_object_list)
        num_matches = 0
        sublist = []
        for item in range(item_count):
            if ai_object_list[item].type & type_mask:
                if ai_object_list[item].id == id or MATCH_ALL_ID == id:
                    num_matches += 1
                    #sort objects by object area in descending order
                    current_object_area = ai_object_list[item].height * ai_object_list[item].width
                    current_object_smallest = True
                    for i in range(len(sublist)):
                            if current_object_area >= (sublist[i].width * sublist[i].height):
                                sublist.insert(i, ai_object_list[item]) # insert item at position i of sublist.
                                current_object_smallest = False
                                break
                    if current_object_smallest:
                        sublist.append(ai_object_list[item]) #add to the end
            
        if num_matches > count:
            num_matches = count

        self._object_count_val = num_matches
        if sublist:
            self._largest_object = sublist[0]
        else:
            self._largest_object = None
        return sublist[:num_matches]
    
    def largest_object(self):
        '''### Request the largest object from the last take_snapshot(...) call

        #### Arguments:
            None

        #### Returns:
            An AiVisionObject object or None if it does not exist
        '''        
        return self._largest_object

    def object_count(self):
        '''### Request the number of objects found in the last take_snapshot call

        #### Arguments:
            None

        #### Returns:
            The number of objects found in the last take_snapshot call
        '''
        return self._object_count_val

    
    def tag_detection(self, enable: bool):
        '''### Enable or disable apriltag processing

        #### Arguments:
            enable : True or False

        #### Returns:
            None
        '''
        json_msg = {"cmd_id": "tag_detection", "b_enable": enable}
        self.robot_instance.robot_send(json_msg)
        pass

    def color_detection(self, enable: bool, merge: bool = False):
        '''### Enable or disable color and code object processing

        #### Arguments:
            enable : True or False
            merge (optional) : True to enable merging of adjacent color detections

        #### Returns:
            None
        '''
        json_msg = {"cmd_id": "color_detection", "b_enable": enable, "b_merge": merge}
        self.robot_instance.robot_send(json_msg)
        pass

    def model_detection(self, enable: bool):
        '''### Enable or disable AI model object processing

        #### Arguments:
            enable : True or False

        #### Returns:
            None
        '''
        json_msg = {"cmd_id": "model_detection", "b_enable": enable}
        self.robot_instance.robot_send(json_msg)
        pass

    def has_barrel(self):
        """returns true if a barrel is held by the kicker"""
        ai_objects = list(self.take_snapshot(AiVision.ALL_AIOBJS))
        has_barrel = False
        for object in range(len(ai_objects)):
            cx = ai_objects[object].originX + ai_objects[object].width/2
            if ai_objects[object].classname in ["BlueBarrel", "OrangeBarrel"] and \
               BARREL_MIN_CX < cx < BARREL_MAX_CX and \
               ai_objects[object].originY > BARREL_MIN_Y :
               has_barrel = True
        return has_barrel

# leave magic numbers same as barrel for now
    def has_ball(self):
        """returns true if a ball is held by the kicker"""
        ai_objects = list(self.take_snapshot(AiVision.ALL_AIOBJS))
        has_ball = False
        for object in range(len(ai_objects)):
            cx = ai_objects[object].originX + ai_objects[object].width/2
            if ai_objects[object].classname in ["Ball"] and \
               BARREL_MIN_CX < cx < BARREL_MAX_CX and \
               ai_objects[object].originY > BARREL_MIN_Y :
               has_ball = True
        return has_ball

class AiVisionObject():
    def __init__(self):
        self.type      = 0
        self.id        = 0
        self.originX   = 0
        self.originY   = 0
        self.width     = 0
        self.height    = 0
        self.classname = ''

class Screen():
    def __init__(self, robot_instance: Robot):
        self.robot_instance = robot_instance
        pass

    def print_at(self, string, x=0, y=0, opaque=True):
        json_cmd = {
            "cmd_id"   : "lcd_print_at",
            "x"        : x,
            "y"        : y,
            "b_opaque" : opaque,
            "string"   : string
        }
        self.robot_instance.robot_send(json_cmd)
    
    def clear_screen( self, color=vex.Color.BLACK ):
        if isinstance(color, (vex.Color, vex.Color.DefinedColor)):
            r = (color.value >> 16) & 0xFF
            g = (color.value >>  8) & 0xFF
            b =  color.value & 0xFF
        else:
            raise aim_exception("clear_screen parameter must be a Color instance")
        json_cmd = {
            "cmd_id"   : "lcd_clear_screen",
            "r"        : r,
            "g"        : g,
            "b"        : b
        }
        self.robot_instance.robot_send(json_cmd)

    def show_emoji(self, emoji: vex.Emoji.EmojiType, look: vex.EmojiLook.EmojiLookType = vex.EmojiLook.CENTER):
        """Show an emoji from a list of preset emojis"""
        json = {
            "cmd_id": "show_emoji",
            "name"  : emoji.value,
            "look"  : look.value
            }
        self.robot_instance.robot_send(json)

    def hide_emoji(self):
        """hide emoji from being displayed, so that any underlying graphics can be viewed"""
        json = {"cmd_id": "hide_emoji"}
        self.robot_instance.robot_send(json)
     
