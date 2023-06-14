import io
import pickle


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module

        # Ensure old checkpoints can still be loaded
        renamed_module = renamed_module.replace(
            "simnet.shape_pretraining_articulated", "CARTO.Decoder"
        )
        renamed_module = renamed_module.replace("simnet.lib", "CARTO.simnet.lib")
        return super(Unpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return Unpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)
