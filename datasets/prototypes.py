import h5py


class Prototype:
    def __init__(self, proto_file):
        self.protos = h5py.File(proto_file, "r")
