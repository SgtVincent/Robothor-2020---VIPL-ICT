import flask
import optparse
import tornado.wsgi
import tornado.httpserver
from datetime import timedelta
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from flask import request
import time

import os
import sys
import io
import json

sys.path.append('../')
sys.path.append('.')

from demo_controller import DemoController
from PIL import Image

# from birdview import get_birdview

DEMO_ONLINE = True
# controller = DemoController(offline_root='/Users/wuxiaodong/data/Robothor_data', verbose=True, scene='FloorPlan_Train1_1')
controller = DemoController(offline_root='/home/ubuntu/Robothor_data/', verbose=True, scene='FloorPlan_Train10_1')

# Obtain the flask app object
app = flask.Flask(__name__, static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
import os


def pil2datauri(img):
    # converts PIL image to datauri
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    img_base64 = bytes("data:image/jpeg;base64, ", encoding='utf-8') + img_str
    return img_base64


#
# def pil2datauri(img):
#     # converts PIL image to datauri
#     buffered = BytesIO()
#     img.save(buffered, format="JPEG")
#     encoded_img = base64.b64encode(buffered.getvalue()).decode('ascii')
#     # img_base64 = bytes("data:image/jpeg;base64,", encoding='utf-8') + img_str
#     return encoded_img

# def get_byte_image(pil_image):
#     img_byte_arr = BytesIO()
#     pil_image.save(img_byte_arr, format='PNG')
#     encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
#     return encoded_img

# server side code
# image_path = 'images/test.png'
# image = get_byte_image(image_path)
# response =  { 'Status' : 'Success', 'message': message, 'ImageBytes': image}


@app.route('/takeaction/', methods=["GET", "POST"])
def take_action():
    global controller
    try:
        time.sleep(0.2)
        action = request.form['action']
        # action = request.args.get('action')
        print(action)
        if action == 'MoveAhead':
            frame = controller.step(action='MoveAhead', online=DEMO_ONLINE)
        elif action == 'MoveBack':
            frame = controller.step(action='MoveBack', online=DEMO_ONLINE)
        elif action == 'RotateLeft':
            frame = controller.step(action='RotateLeft', online=DEMO_ONLINE)
        elif action == 'RotateRight':
            frame = controller.step(action='RotateRight', online=DEMO_ONLINE)
        elif action == 'LookUp':
            frame = controller.step(action='LookUp', online=DEMO_ONLINE)
        elif action == 'LookDown':
            frame = controller.step(action='LookDown', online=DEMO_ONLINE)
        elif action == 'GetCurrentFrame':
            frame = controller.step(action='GetCurrentFrame', online=DEMO_ONLINE)
        else:
            pass
        img_str = pil2datauri(frame)
    except:
        pass

    response = {'robo_state': controller.scene + '        ' + controller.current_state(online=DEMO_ONLINE),
                'ImageBytes': img_str.decode('ascii')}
    # return flask.jsonify(response)
    return json.dumps(response)


@app.route('/teleport', methods=["POST"])
def teleport():
    try:
        global DEMO_ONLINE
        global controller
        state = flask.request.form['state']
        success = controller.teleport_to_state(state, online=DEMO_ONLINE)
    except:
        pass
    return flask.render_template('index.html')


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/online/<scene>')
def online_demo(scene):
    try:
        global DEMO_ONLINE
        global controller
        controller.reset(scene)
        DEMO_ONLINE = True
    except:
        pass
    return flask.render_template('index.html')


@app.route('/offline/<scene>')
def offline_demo(scene):
    try:
        global DEMO_ONLINE
        global controller
        DEMO_ONLINE = False
        controller.reset(scene, online=DEMO_ONLINE)
    except:
        pass
    return flask.render_template('index.html')


# @app.route('/<scene>')
# def (scene):
#     global controller
#     controller.reset(scene)
#     return flask.render_template('index.html')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=6002
    )
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=opts.port)
    # logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)
