import math

from ai2thor.controller import Controller

from PIL import Image, ImageDraw

import copy
import numpy as np
import os

FRAME_SHAPE = (300, 300, 3)
CAM_POSITION = (5.05, 7.614471, -5.663423)
ORTH_SIZE = 6.261584

BIRD_VIEW_ROOT = './data/birdview'


class ThorPositionTo2DFrameTranslator(object):
    def __init__(self):
        self.frame_shape = FRAME_SHAPE
        self.lower_left = np.array((CAM_POSITION[0], CAM_POSITION[2])) - ORTH_SIZE
        self.span = 2 * ORTH_SIZE

    def __call__(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position

        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )


def add_agent_view_triangle(
        position, rotation, frame, scale=1, opacity=0.7
):
    p0 = np.array((position[0], position[2]))
    p1 = copy.copy(p0)
    p2 = copy.copy(p0)

    theta = -2 * math.pi * (rotation / 360.0)
    rotation_mat = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    offset1 = scale * np.array([-1, 1]) * math.sqrt(2) / 2
    offset2 = scale * np.array([1, 1]) * math.sqrt(2) / 2

    p1 += np.matmul(rotation_mat, offset1)
    p2 += np.matmul(rotation_mat, offset2)

    img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
    # img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA
    img2 = Image.new("RGBA", img1.size)  # Use RGBA

    opacity = int(round(255 * opacity))  # Define transparency for the triangle.
    points = [tuple(reversed(pos_translator(p))) for p in [p0, p1, p2]]
    draw = ImageDraw.Draw(img2)
    draw.polygon(points, fill=(255, 255, 255, opacity))
    # import ipdb
    # ipdb.set_trace()
    img = Image.alpha_composite(img1, img2)
    return np.array(img.convert("RGB"))


pos_translator = ThorPositionTo2DFrameTranslator()


def get_trajectory(scene_name, loc_strs, birdview_root, init_loc_str='', target_loc_str='', actions=[], success=True,
                   target_name=''):
    frame_pil = Image.open(os.path.join(birdview_root, '{}.png'.format(scene_name)))
    frame = np.asarray(frame_pil)
    image_points = []

    # add init rotation indicator
    x, z, rotation, horizon = [float(x) for x in init_loc_str.split("|")]
    frame = add_agent_view_triangle(
        position=(x, 0, z),
        rotation=rotation,
        frame=frame,
    )

    # add the rotation indicator of the last action
    x, z, rotation, horizon = [float(x) for x in loc_strs[-1].split("|")]
    frame = add_agent_view_triangle(
        position=(x, 0, z),
        rotation=rotation,
        frame=frame,
    )
    frame_pil = Image.fromarray(frame)
    for loc_str in loc_strs:
        x, z, rotation, horizon = [float(x) for x in loc_str.split("|")]
        p_x, p_y = pos_translator((x, z))
        image_points.append((p_x, p_y))
        # frame = add_agent_view_triangle(
        #     position=(x, 0, z),
        #     rotation=rotation,
        #     frame=frame,
        #     image_point=(p_x, p_y),
        #     init_loc_str=init_loc_str,
        #     target_loc_str=target_loc_str
        # )

        # draw.rectangle((x, 100, 300, 200), fill=(0, 192, 192), outline=(255, 255, 255))
        # draw.line((350, 200, 450, 100), fill=(255, 255, 0), width=10)

        # draw_im = ImageDraw.Draw(frame_pil)
        # draw_im.ellipse((200, 2000, 300, 300), fill=(255, 0, 0))
        # # img = Image.fromarray(new_frame)
        # frame_pil.show()

    # frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)

    # plot trajectory
    if len(image_points) >= 2:
        for i in range(len(image_points) - 1):
            start_p = image_points[i]
            end_p = image_points[i + 1]
            # draw.line((start_p[1], start_p[0], end_p[1], end_p[0]), fill=(255, 255, 0), width=2)
            draw.line((start_p[1], start_p[0], end_p[1], end_p[0]), fill=(0, 0, 255), width=4)

    # plot init and target point
    if init_loc_str != '' and target_loc_str != '':
        init_x, init_x_z, init_rotation, init_horizon = [float(x) for x in init_loc_str.split("|")]
        init_p_x, init_p_y = pos_translator((init_x, init_x_z))
        draw.ellipse((init_p_y - 5, init_p_x - 5, init_p_y + 5, init_p_x + 5), fill=(255, 0, 0), outline=(0, 0, 0))

        target_x, target_x_z, target_rotation, target_horizon = [float(x) for x in target_loc_str.split("|")]
        target_p_x, target_p_y = pos_translator((target_x, target_x_z))
        draw.rectangle((target_p_y - 5, target_p_x - 5, target_p_y + 5, target_p_x + 5), fill=(255, 0, 0),
                       outline=(0, 0, 0))
    if len(actions) != 0:
        for idx, action in enumerate(actions):
            if action.startswith('Move'):
                action = action[4:]
            elif action.startswith('Rotate'):
                action = action[6:]
            elif action.startswith('Look'):
                action = action[4:]
            else:
                pass
            left = idx // 13
            top = idx % 13
            draw.text([30 + 40 * left, 160 + 10 * top], action, 'green' if success else 'red')

    # draw legend
    draw.ellipse((5, 5, 15, 15), fill=(255, 0, 0), outline=(0, 0, 0))
    draw.rectangle((5, 15, 15, 25), fill=(255, 0, 0), outline=(0, 0, 0))
    draw.text([20, 5], 'Init Agent', 'red')
    draw.text([20, 15], 'Target: {}'.format(target_name), 'red')

    # frame_pil.save('test.png')
    # import ipdb
    # ipdb.set_trace()
    return frame_pil
    # frame_pil.show()
    # pass


def get_agent_map_data(c: Controller):
    c.step({"action": "ToggleMapView"})
    cam_position = c.last_event.metadata["cameraPosition"]
    cam_orth_size = c.last_event.metadata["cameraOrthSize"]
    pos_translator = ThorPositionTo2DFrameTranslator(
        c.last_event.frame.shape, position_to_tuple(cam_position), cam_orth_size
    )
    to_return = {
        "frame": c.last_event.frame,
        "cam_position": cam_position,
        "cam_orth_size": cam_orth_size,
        "pos_translator": pos_translator,
    }
    c.step({"action": "ToggleMapView"})
    return to_return


def get_birdview(c):
    t = get_agent_map_data(c)
    new_frame = add_agent_view_triangle(
        position_to_tuple(c.last_event.metadata["agent"]["position"]),
        c.last_event.metadata["agent"]["rotation"]["y"],
        t["frame"],
        t["pos_translator"],
    )
    return new_frame


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # import matplotlib

    # matplotlib.use("TkAgg", warn=False)
    # c = Controller(width=640, height=480, rotateStepDegrees=30, applyActionNoise=False, gridSize=0.125,
    #                snapToGrid=False, agentMode='bot')
    # print(c.last_event.metadata["agent"]["position"])
    # print(c.last_event.metadata["cameraPosition"])

    # event = c.step(action='TeleportFull', x=1.25, y=7.614471, z=-4.25,
    #                                     rotation=dict(x=0.0, y=120, z=0.0), horizon=-30)

    # c.start()

    """
    generate birdview image
    """
    # import os
    # from PIL import Image
    #
    # c = Controller()
    #
    # bird_view_root = './data/birdview'cd
    # bird_view_root = './data/birdview'cd
    # if not os.path.exists(bird_view_root):
    #     os.makedirs(bird_view_root)
    # for i in range(12):
    #     for j in range(5):
    #         scene_name = "FloorPlan_Train{}_{}".format(i + 1, j + 1)
    #         c.reset(scene_name)
    #         c.step({"action": "ToggleMapView"})
    #         frame = Image.fromarray(c.last_event.frame)
    #         frame.save(os.path.join(bird_view_root, scene_name + '.png'))

    # t = get_agent_map_data(c)
    # new_frame = add_agent_view_triangle(
    #     position_to_tuple(c.last_event.metadata["agent"]["position"]),
    #     c.last_event.metadata["agent"]["rotation"]["y"],
    #     t["frame"],
    #     t["pos_translator"],
    # )
    # new_frame = get_birdview(c)
    # plt.imshow(new_frame)
    # plt.show()
    get_trajectory('FloorPlan_Train1_1', ['1.250|-4.250|0|-30', '2.125|-3.000|270|30'], birdview_root=BIRD_VIEW_ROOT,
                   init_loc_str='1.250|-4.250|0|-30', target_loc_str='2.125|-3.000|270|30')