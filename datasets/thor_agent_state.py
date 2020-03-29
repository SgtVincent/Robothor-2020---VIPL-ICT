class ThorAgentState:
    """ Representation of a simple state of a Thor Agent which includes
        the position, horizon and rotation. """

    def __init__(self, x, y, z, rotation, horizon, state_decimal=2):
        self.x = round(x, state_decimal)
        self.y = y
        self.z = round(z, state_decimal)
        self.rotation = round(rotation)
        self.horizon = round(horizon)
        self.state_decimal = state_decimal

    @classmethod
    def get_state_from_evenet(cls, event, forced_y=None, state_decimal=2):
        """ Extracts a state from an event. """
        state = cls(
            x=event.metadata["agent"]["position"]["x"],
            y=event.metadata["agent"]["position"]["y"],
            z=event.metadata["agent"]["position"]["z"],
            rotation=event.metadata["agent"]["rotation"]["y"],
            horizon=event.metadata["agent"]["cameraHorizon"],
            state_decimal=state_decimal
        )
        if forced_y != None:
            state.y = forced_y
        return state

    def __eq__(self, other):
        """ If we check for exact equality then we get issues.
            For now we consider this 'close enough'. """
        if isinstance(other, ThorAgentState):
            return (
                self.x == other.x
                and
                # self.y == other.y and
                self.z == other.z
                and self.rotation == other.rotation
                and self.horizon == other.horizon
            )
        return NotImplemented

    def __str__(self):
        """ Get the string representation of a state. """
        """
        return '{:0.2f}|{:0.2f}|{:0.2f}|{:d}|{:d}'.format(
            self.x,
            self.y,
            self.z,
            round(self.rotation),
            round(self.horizon)
        )
        """
        # Use args.state_decimal to control the decimal of state key
        output_pattern = "{:0."+ str(self.state_decimal) + "f}|{:0." + str(self.state_decimal) + "f}|{:d}|{:d}"

        return output_pattern.format(
            self.x, self.z, round(self.rotation), round(self.horizon)
        )

    def position(self):
        """ Returns just the position. """
        return dict(x=self.x, y=self.y, z=self.z)
