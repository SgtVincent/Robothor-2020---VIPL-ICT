############# Ithor Scenes Constants #####################
KITCHEN_OBJECT_CLASS_LIST = [
    "Toaster",
    "Microwave",
    "Fridge",
    "CoffeeMaker",
    "GarbageCan",
    "Box",
    "Bowl",
]

LIVING_ROOM_OBJECT_CLASS_LIST = [
    "Pillow",
    "Laptop",
    "Television",
    "GarbageCan",
    "Box",
    "Bowl",
]

BEDROOM_OBJECT_CLASS_LIST = ["HousePlant", "Lamp", "Book", "AlarmClock"]

BATHROOM_OBJECT_CLASS_LIST = ["Sink", "ToiletPaper", "SoapBottle", "LightSwitch"]

FULL_OBJECT_CLASS_LIST = (
        KITCHEN_OBJECT_CLASS_LIST
        + LIVING_ROOM_OBJECT_CLASS_LIST
        + BEDROOM_OBJECT_CLASS_LIST
        + BATHROOM_OBJECT_CLASS_LIST
)

################# Robothor Scenes Constants ##################

ROBOTHOR_ORIGINAL_CLASS_LIST = ['AlarmClock', 'Apple', 'BaseballBat', 'BasketBall', 'Bowl', 'GarbageCan', 'HousePlant',
                                'Laptop', 'Mug', 'SprayBottle', 'Television', 'Vase']

################# General Constants ###########################

MOVE_AHEAD = "MoveAhead"
ROTATE_LEFT = "RotateLeft"
ROTATE_RIGHT = "RotateRight"
LOOK_UP = "LookUp"
LOOK_DOWN = "LookDown"
MOVE_BACK = "MoveBack"
STOP = "Stop"
DONE = "Done"


# DONE_ACTION_INT = 5
# GOAL_SUCCESS_REWARD = 5
# STEP_PENALTY = -0.01
