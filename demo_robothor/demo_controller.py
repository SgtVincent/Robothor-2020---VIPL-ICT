# encoding: utf-8
"""
@author: Xiaodong Wu
@version: 1.0
@file: demo_controller.py
@time:  2020-04-08 17:53

"""
from PIL import Image
import numpy as np
from ai2thor.controller import Controller
from birdview import get_birdview
from datasets.offline_controller_with_small_rotation import OfflineControllerWithSmallRotation


def get_frame(c, show_bird_view=False):
    frame = c.last_event.frame

    if show_bird_view:
        bird_view = get_birdview(c)
        new_frame = np.concatenate((frame, bird_view), axis=1)
        return new_frame
    else:
        return frame


class DemoController():
    def __init__(self, offline_root='', scene='FloorPlan_Train1_1', verbose=False):
        self.verbose = verbose
        self.online_controller = Controller(width=640,
                                            height=480,
                                            rotateStepDegrees=30,
                                            applyActionNoise=False,
                                            gridSize=0.125,
                                            snapToGrid=False,
                                            agentMode='bot')

        # self.online_controller = ai2thor.controller.Controller(
        #     start_unity=True,
        #     width=640,
        #     height=480,
        #     agentMode='bot',
        #     gridSize=0.125,
        #     rotateStepDegrees=30,
        #     applyActionNoise=False,
        #     snapToGrid=False)
        self.scene = scene
        if offline_root != '':
            self.offline_controller = OfflineControllerWithSmallRotation(
                grid_size=0.125,
                fov=90,
                offline_data_dir=offline_root,
                visualize=False,
                rotate_by=30,
                state_decimal=3
            )
        else:
            self.offline_controller = None

        if scene != '':
            self.online_controller.reset(scene)
            if self.offline_controller is not None:
                self.offline_controller.reset(scene)

    # todo
    def step(self, action, online=True):
        if online:
            if action == 'GetCurrentFrame':
                event = self.online_controller.last_event
            else:
                event = self.online_controller.step(action=action)
            scene_frame = Image.fromarray(event.frame)
            if self.verbose:
                self.log_message(self.online_controller)
            brid_frame = get_birdview(self.online_controller)
            frame = np.concatenate((scene_frame, brid_frame), axis=1)

        else:
            if self.offline_controller is not None:
                if action == 'GetCurrentFrame':
                    return Image.fromarray(self.offline_controller.get_image())
                event = self.offline_controller.step(action={'action': action})
                frame = self.offline_controller.last_event.frame
                # if self.verbose:
                #     The first action will have no attribute of 'lastAction' which will raise error
                # self.log_message(self.offline_controller)
        return Image.fromarray(frame)

    def teleport_to_state(self, state_str, online=True):
        """ Only use this method when we know the state is valid. """
        teleport_success = False
        y = 0.9009997
        # 1.250|-2.125|210|0
        x, z, rotation_degree, horizon_degree = [round(float(k), 3) for k in state_str.split("|")]
        if online:
            event = self.online_controller.step(action='TeleportFull', x=x, y=y, z=z,
                                                rotation=dict(x=0.0, y=rotation_degree, z=0.0), horizon=horizon_degree)

            # viz_event = self.online_controller.step(
            #     dict(action="Teleport", x=x, y=y, z=z)
            # )
            # if not viz_event.metadata["lastActionSuccess"]:
            #     teleport_success = False
            #     return teleport_success
            #
            # viz_event = self.online_controller.step(
            #     dict(action="Rotate", rotation=rotation_degree)
            # )
            # if not viz_event.metadata["lastActionSuccess"]:
            #     teleport_success = False
            #     return teleport_success
            #
            # viz_event = self.online_controller.step(
            #     dict(action="Look", horizon=horizon_degree)
            # )
            # if not viz_event.metadata["lastActionSuccess"]:
            #     teleport_success = False
            #     return teleport_success
            teleport_success = event.metadata["lastActionSuccess"]
        else:
            if state_str in self.offline_controller.images:
                teleport_success = True
                self.offline_controller.state = self.offline_controller.get_state_from_str(x, z,
                                                                                           rotation=rotation_degree,
                                                                                           horizon=horizon_degree)
            else:
                teleport_success = False
        return teleport_success

    def reset(self, scene, online=True):
        self.scene = scene
        self.online_controller.reset(scene)
        if self.offline_controller is not None and not online:
            self.offline_controller.reset(scene)
        return

    def log_message(self, c):
        print('.......')
        print('lastAction: ', c.last_event.metadata['lastAction'])
        print('lastActionSuccess: ', c.last_event.metadata['lastActionSuccess'])
        print('errorMessage: ', c.last_event.metadata['errorMessage'])
        print('position: ', c.last_event.metadata["agent"]["position"])
        print('rotation: ', c.last_event.metadata["agent"]["rotation"])
        print('\n')

    @property
    def online_frame(self):
        return Image.fromarray(self.online_controller.last_event.frame)

    @property
    def offline_frame(self):
        if self.offline_controller is not None:
            return Image.fromarray(self.offline_controller.get_image())
            # return Image.fromarray(self.offline_controller.last_event.metadata['lastAction'])

    def get_current_frame(self, online=True):
        if online:
            return self.online_frame
        else:
            return self.offline_frame
        return

    def current_state(self, online=True):
        output_pattern = "{:0.3f}|{:0.3f}|{:d}|{:d}"

        if online:
            x = self.online_controller.last_event.metadata["agent"]["position"]['x']
            z = self.online_controller.last_event.metadata["agent"]["position"]['z']
            rotation = self.online_controller.last_event.metadata["agent"]["rotation"]['y']
            horizon = self.online_controller.last_event.metadata["agent"]["rotation"]['z']
        else:
            x = self.offline_controller.state.x
            z = self.offline_controller.state.z
            rotation = self.offline_controller.state.rotation
            horizon = self.offline_controller.state.horizon
        return "{:0.3f}|{:0.3f}|{:d}|{:d}".format(x, z, int(rotation + 0.5), int(horizon+0.5))


if __name__ == '__main__':
    demo_c = DemoController(offline_root='/Users/wuxiaodong/data/Robothor_data', verbose=True)
    # Image.fromarray(demo_c.online_frame).show()
    Image.fromarray(demo_c.step('MoveAhead', online=False)).show()
    Image.fromarray(demo_c.step('MoveAhead', online=False)).show()
    Image.fromarray(demo_c.step('MoveAhead', online=False)).show()
