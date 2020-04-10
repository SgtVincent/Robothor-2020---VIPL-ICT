from datasets.constants import (
    MOVE_AHEAD,
    MOVE_BACK,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_UP,
    LOOK_DOWN,
    DONE,
)

# modified on 2020/04/08 by junting
# WARNING: DONE action should be put at the end of the list
def get_actions(args):
    if args.action_space == 6:
        return [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_UP, LOOK_DOWN, DONE]
    elif args.action_space == 4:
        return [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, DONE]
    elif args.action_space == 7:
        return [MOVE_AHEAD, MOVE_BACK, ROTATE_LEFT, ROTATE_RIGHT, LOOK_UP, LOOK_DOWN, DONE]
    else:
        raise Exception("Expected number of possible actions: 4,6,7.")

