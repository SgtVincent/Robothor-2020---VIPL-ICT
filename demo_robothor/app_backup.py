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

import os
import sys

sys.path.append('../')
sys.path.append('.')

import ai2thor.controller
from PIL import Image
# from birdview import get_birdview

ONLINE_DEMO = True
controller = ai2thor.controller.Controller(
    start_unity=True,
    width=640,
    height=480,
    scene='FloorPlan_Train12_1',
    agentMode='bot',
    gridSize=0.125,
    rotateStepDegrees=30,
    applyActionNoise=False,
    snapToGrid=False)


# Obtain the flask app object
app = flask.Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
import os


def pil2datauri(img):
    # converts PIL image to datauri
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    img_base64 = bytes("data:image/jpeg;base64,", encoding='utf-8') + img_str
    return img_base64



@app.route('/takeaction/', methods=["GET", "POST"])
def take_action():
    action = request.args.get('action')
    print(action)
    if action == 'MoveAhead':
        event = controller.step(action='MoveAhead')
    elif action == 'MoveBack':
        event = controller.step(action='MoveBack')
    elif action == 'RotateLeft':
        event = controller.step(action='RotateLeft')
    elif action == 'RotateRight':
        event = controller.step(action='RotateRight')
    elif action == 'LookUp':
        event = controller.step(action='LookUp')
    elif action == 'LookDown':
        event = controller.step(action='LookDown')
    else:
        pass

    # event = controller.step(
    #     dict(action="Teleport", x=1.25, y=0.9009997, z=-1.25)
    # )
    # event = controller.step(dict(action="Rotate", rotation=120))
    img = Image.fromarray(event.frame)
    # imarray = np.random.rand(100, 100, 3) * 255
    # img = Image.fromarray(imarray.astype('uint8'))
    img_str = pil2datauri(img)
    return img_str


@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/online')
def online_demo():
    ONLINE_DEMO = True
    return flask.render_template('index.html')

@app.route('/offline')
def offline_demo():
    ONLINE_DEMO = False
    return flask.render_template('index.html')

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
        type='int', default=5000)
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
