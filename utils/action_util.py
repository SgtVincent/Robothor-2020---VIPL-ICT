from datasets.constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_UP,
    LOOK_DOWN,
    DONE,
)


# def get_actions(args):
#     assert args.action_space == 6, "Expected 6 possible actions."
#     return [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_UP, LOOK_DOWN, DONE]

# modified on 2019/12/17 by junting
def get_actions(args):
    if args.action_space == 6:
        return [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_UP, LOOK_DOWN, DONE]
    elif args.action_space == 4:
        return [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, DONE]
    else:
        print("Expected number of possible actions: 4,6.")
