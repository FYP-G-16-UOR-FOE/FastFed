import io
import pickle


class Serializer:

    @staticmethod
    def serialize(data):
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        return buffer.getvalue()
    
    @staticmethod
    def deserialize(serialized_state):
        buffer = io.BytesIO(serialized_state)
        deserialized_state = pickle.load(buffer)
        return deserialized_state