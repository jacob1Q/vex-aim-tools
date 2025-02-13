from aim_fsm import *

new_preamble = """
  You are an intelligent mobile robot named Celeste.
  You have a plastic cylindrical body with a diameter of 65 mm and a height of 72 mm.
  You have three omnidirectional wheels and a forward-facing camera.
  You converse with humans and answer questions as concisely as possible.
  Here is how to control your body:
  To move forward by N millimeters, output the string "#forward N" without quotes.
  To move to the left by N milllimeters, output the string "#sideways N" without quotes, and use a negative value to move right.
  To turn counter-clockwise by N degrees, output the string "#turn N" without quotes, and use a negative value for clockwise turns.
  To turn toward object X, output the string "#turntoward X" without quotes.
  To pick up object X, output the string "#pickup X" without quotes.
  To drop an object, output the string "#drop" without quotes.
  To obtain the current camera image, output the string '#camera" without quotes.
  When using any of these # commands, the command must appear on a line by itself, with nothing preceding it.
  When asked what you see in the camera, first obtain the current camera image, then answer the question after receiving the image.
  Pronounce "AprilTag-1.a" as "April Tag 1-A", and similarly for any word of form "AprilTag-N.x".
  Pronounce "OrangeBarrel.a" as "Orange Barrel A", pronounce "BlueBarrel.b" as "Blue Barrel B", and similarly for other barrel designators.
  Remember to be concise in your answers.
"""

class GPT_test(StateMachineProgram):

    class CheckResponse(StateNode):
        def start(self, event):
            super().start(event)
            response_string = event.response
            lines = list(filter(lambda x: len(x)>0, response_string.split('\n')))
            # If the response contains any #command lines then convert
            # raw text lines to #say commands.
            if any((line.startswith('#') for line in lines)):
                commands = [line if line.startswith('#') else ('#say ' + line) for line in lines]
                print(commands)
                self.post_data(commands)
            # else response is a pure string so just speak it in one gulp
            else:
                self.post_data(response_string)

    class CmdForward(Forward):
      def start(self,event):
          print(event.data)
          self.distance_mm = float((event.data.split(' '))[1])
          super().start(event)

    class CmdSideways(Sideways):
      def start(self,event):
          print(event.data)
          self.distance_mm = float((event.data.split(' '))[1])
          super().start(event)

    class CmdTurn(Turn):
      def start(self,event):
          print(event.data)
          self.angle_deg = float((event.data.split(' '))[1])
          super().start(event)

    class CmdTurnToward(TurnToward):
        def start(self,event):
            print(event.data)
            spec = event.data.split(' ')
            self.obj_spec  = ''.join(spec[1:])
            print('Turning toward', self.object_spec)
            super().start(None)

    class CmdPickup(PickUp):
      def start(self,event):
          print(event.data)
          spec = event.data.split(' ')
          self.object_spec = ''.join(spec[1:])
          print('Picking up', self.object_spec)
          super().start(None)

    class CmdDrop(Drop):
      def start(self,event):
          print(event.data)
          super().start(event)

    class CmdSendCamera(SendGPTCamera):
        def start(self,event):
            print(event.data)
            super().start(event)

    class CmdSay(Say):
        def start(self,event):
            print('#say ...')
            self.text = event.data[5:]
            super().start(event)

    class SpeakResponse(Say):
      def start(self,event):
        self.text = event.data
        super().start(event)
        
    def start(self):
        self.robot.openai_client.set_preamble(new_preamble)
        super().start()

    def setup(self):
        #       Say("Talk to me") =C=> loop
        # 
        #       loop: StateNode() =Hear()=> AskGPT() =OpenAITrans()=> check
        # 
        #       check: self.CheckResponse()
        #       check =D(list)=> dispatch
        #       check =D(str)=> self.SpeakResponse() =C=> loop
        # 
        #       dispatch: Iterate()
        #       dispatch =D(re.compile('#say '))=> self.CmdSay() =CNext=> dispatch
        #       dispatch =D(re.compile('#forward '))=> self.CmdForward() =CNext=> dispatch
        #       dispatch =D(re.compile('#sideways '))=> self.CmdSideways() =CNext=> dispatch
        #       dispatch =D(re.compile('#turn '))=> self.CmdTurn() =CNext=> dispatch
        #       dispatch =D(re.compile('#turntoward '))=> turntoward
        #       dispatch =D(re.compile('#drop$'))=> self.CmdDrop() =CNext=> dispatch
        #       dispatch =D(re.compile('#pickup '))=> pickup
        #       dispatch =D(re.compile('#camera$'))=> self.CmdSendCamera() =C=>
        #         AskGPT("Please respond to the query using the camera image.") =OpenAITrans()=> check
        #       dispatch =D()=> Print() =Next=> dispatch
        #       dispatch =C=> loop
        # 
        #       turntoward: self.CmdTurnToward()
        #       turntoward =CNext=> dispatch
        #       turntoward =F=> StateNode() =Next=> dispatch
        # 
        #       pickup: self.CmdPickup()
        #       pickup =CNext=> dispatch
        #       pickup =F=> StateNode() =Next=> dispatch
        # 
        
        # Code generated by genfsm on Thu Feb 13 14:31:27 2025:
        
        say1 = Say("Talk to me") .set_name("say1") .set_parent(self)
        loop = StateNode() .set_name("loop") .set_parent(self)
        askgpt1 = AskGPT() .set_name("askgpt1") .set_parent(self)
        check = self.CheckResponse() .set_name("check") .set_parent(self)
        speakresponse1 = self.SpeakResponse() .set_name("speakresponse1") .set_parent(self)
        dispatch = Iterate() .set_name("dispatch") .set_parent(self)
        cmdsay1 = self.CmdSay() .set_name("cmdsay1") .set_parent(self)
        cmdforward1 = self.CmdForward() .set_name("cmdforward1") .set_parent(self)
        cmdsideways1 = self.CmdSideways() .set_name("cmdsideways1") .set_parent(self)
        cmdturn1 = self.CmdTurn() .set_name("cmdturn1") .set_parent(self)
        cmddrop1 = self.CmdDrop() .set_name("cmddrop1") .set_parent(self)
        cmdsendcamera1 = self.CmdSendCamera() .set_name("cmdsendcamera1") .set_parent(self)
        askgpt2 = AskGPT("Please respond to the query using the camera image.") .set_name("askgpt2") .set_parent(self)
        print1 = Print() .set_name("print1") .set_parent(self)
        turntoward = self.CmdTurnToward() .set_name("turntoward") .set_parent(self)
        statenode1 = StateNode() .set_name("statenode1") .set_parent(self)
        pickup = self.CmdPickup() .set_name("pickup") .set_parent(self)
        statenode2 = StateNode() .set_name("statenode2") .set_parent(self)
        
        completiontrans1 = CompletionTrans() .set_name("completiontrans1")
        completiontrans1 .add_sources(say1) .add_destinations(loop)
        
        heartrans1 = HearTrans() .set_name("heartrans1")
        heartrans1 .add_sources(loop) .add_destinations(askgpt1)
        
        openaitrans1 = OpenAITrans() .set_name("openaitrans1")
        openaitrans1 .add_sources(askgpt1) .add_destinations(check)
        
        datatrans1 = DataTrans(list) .set_name("datatrans1")
        datatrans1 .add_sources(check) .add_destinations(dispatch)
        
        datatrans2 = DataTrans(str) .set_name("datatrans2")
        datatrans2 .add_sources(check) .add_destinations(speakresponse1)
        
        completiontrans2 = CompletionTrans() .set_name("completiontrans2")
        completiontrans2 .add_sources(speakresponse1) .add_destinations(loop)
        
        datatrans3 = DataTrans(re.compile('#say ')) .set_name("datatrans3")
        datatrans3 .add_sources(dispatch) .add_destinations(cmdsay1)
        
        cnexttrans1 = CNextTrans() .set_name("cnexttrans1")
        cnexttrans1 .add_sources(cmdsay1) .add_destinations(dispatch)
        
        datatrans4 = DataTrans(re.compile('#forward ')) .set_name("datatrans4")
        datatrans4 .add_sources(dispatch) .add_destinations(cmdforward1)
        
        cnexttrans2 = CNextTrans() .set_name("cnexttrans2")
        cnexttrans2 .add_sources(cmdforward1) .add_destinations(dispatch)
        
        datatrans5 = DataTrans(re.compile('#sideways ')) .set_name("datatrans5")
        datatrans5 .add_sources(dispatch) .add_destinations(cmdsideways1)
        
        cnexttrans3 = CNextTrans() .set_name("cnexttrans3")
        cnexttrans3 .add_sources(cmdsideways1) .add_destinations(dispatch)
        
        datatrans6 = DataTrans(re.compile('#turn ')) .set_name("datatrans6")
        datatrans6 .add_sources(dispatch) .add_destinations(cmdturn1)
        
        cnexttrans4 = CNextTrans() .set_name("cnexttrans4")
        cnexttrans4 .add_sources(cmdturn1) .add_destinations(dispatch)
        
        datatrans7 = DataTrans(re.compile('#turntoward ')) .set_name("datatrans7")
        datatrans7 .add_sources(dispatch) .add_destinations(turntoward)
        
        datatrans8 = DataTrans(re.compile('#drop$')) .set_name("datatrans8")
        datatrans8 .add_sources(dispatch) .add_destinations(cmddrop1)
        
        cnexttrans5 = CNextTrans() .set_name("cnexttrans5")
        cnexttrans5 .add_sources(cmddrop1) .add_destinations(dispatch)
        
        datatrans9 = DataTrans(re.compile('#pickup ')) .set_name("datatrans9")
        datatrans9 .add_sources(dispatch) .add_destinations(pickup)
        
        datatrans10 = DataTrans(re.compile('#camera$')) .set_name("datatrans10")
        datatrans10 .add_sources(dispatch) .add_destinations(cmdsendcamera1)
        
        completiontrans3 = CompletionTrans() .set_name("completiontrans3")
        completiontrans3 .add_sources(cmdsendcamera1) .add_destinations(askgpt2)
        
        openaitrans2 = OpenAITrans() .set_name("openaitrans2")
        openaitrans2 .add_sources(askgpt2) .add_destinations(check)
        
        datatrans11 = DataTrans() .set_name("datatrans11")
        datatrans11 .add_sources(dispatch) .add_destinations(print1)
        
        nexttrans1 = NextTrans() .set_name("nexttrans1")
        nexttrans1 .add_sources(print1) .add_destinations(dispatch)
        
        completiontrans4 = CompletionTrans() .set_name("completiontrans4")
        completiontrans4 .add_sources(dispatch) .add_destinations(loop)
        
        cnexttrans6 = CNextTrans() .set_name("cnexttrans6")
        cnexttrans6 .add_sources(turntoward) .add_destinations(dispatch)
        
        failuretrans1 = FailureTrans() .set_name("failuretrans1")
        failuretrans1 .add_sources(turntoward) .add_destinations(statenode1)
        
        nexttrans2 = NextTrans() .set_name("nexttrans2")
        nexttrans2 .add_sources(statenode1) .add_destinations(dispatch)
        
        cnexttrans7 = CNextTrans() .set_name("cnexttrans7")
        cnexttrans7 .add_sources(pickup) .add_destinations(dispatch)
        
        failuretrans2 = FailureTrans() .set_name("failuretrans2")
        failuretrans2 .add_sources(pickup) .add_destinations(statenode2)
        
        nexttrans3 = NextTrans() .set_name("nexttrans3")
        nexttrans3 .add_sources(statenode2) .add_destinations(dispatch)
        
        return self
