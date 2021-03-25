import pickle


class Serialization:
    @staticmethod
    def save_obj(obj, name):
        """
        serialization of an object
        :param obj: object to serialize
        :param name: file name to store the object
        """
        with open('/ais/hal9000/jai/autism/pickle/' + name + '.pkl', 'wb') as fout:
            pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)
        # end with
    # end def

    @staticmethod
    def load_obj(name):
        """
        de-serialization of an object
        :param name: file name to load the object from
        """
        with open('/ais/hal9000/jai/autism/pickle/' + name + '.pkl', 'rb') as fout:
            return pickle.load(fout)