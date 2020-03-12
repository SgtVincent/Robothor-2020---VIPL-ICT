from ai2thor.controller import Controller

# Teleport wrong case
wrong_case_config = {
    "scene":"FloorPlan1",
    "gridSize":0.25,
    "fieldOfView":90,
    # "agentMode":"bot",
    "cameraY":1.00
}
# 1.25|1.00|270|
init_position = {
    "x":-1.50,
    "y":1.00,
    "z":1.00
}
target_position = {
    "x": -1.75,
    "y": 1.00,
    "z": 1.00
}


def get_state_from_evenet(event):
    x=event.metadata["agent"]["position"]["x"]
    y=event.metadata["agent"]["position"]["y"]
    z=event.metadata["agent"]["position"]["z"]
    rotation=event.metadata["agent"]["rotation"]["y"]
    horizon=event.metadata["agent"]["cameraHorizon"]
    return (x, y, z, rotation, horizon)

if __name__ == "__main__":

    # initialize ai2thor controller
    controller = Controller(
        scene="FloorPlan_Train1_1", gridSize=0.25,
        fieldOfView=90, cameraY=1.00)

    event = controller.step(action="Teleport", x=2.50, y=1.00, z=-1.25)
    print("Initial state: (x, y, z, rotation, horizon)={}".format(get_state_from_evenet(event)))
    event = controller.step(action="Teleport", x=2.50, y=1.00, z=-1.00)
    print("State after Teleport: (x, y, z, rotation, horizon)={}".format(get_state_from_evenet(event)))
    print("Last Action Success:{}; Error Message:{}".format(
        event.metadata["lastActionSuccess"],
        event.metadata["errorMessage"]))