from datetime import datetime
import os
import re
import sys
sys.path.append('.')

from src import paths

def last_checkpoint():
    model_regex = re.compile(r"model.ckpt-([0-9]+)")
    matches = [model_regex.match(f) for f in os.listdir(paths.MODELS)]
    checkpoints = [m for m in matches if m is not None]

    sorter = lambda c: -int(c.group(1))
    checkpoint = sorted(checkpoints, key=sorter)[0]
    return checkpoint.group(0)

if __name__ == "__main__":
    print("Export beginning: " + datetime.now().isoformat())

    command = []
    command.append("PYTHONPATH=$PYTHONPATH:..:../slim")
    command.append("python " + paths.EXTERNAL_EXPORT_SCRIPT)
    command.append("--input_type='image_tensor'")
    command.append("--pipeline_config_path='{}'".format(paths.MODEL_CONFIG))
    command.append("--trained_checkpoint_prefix='{}'".format(os.path.join(paths.MODELS, last_checkpoint())))
    command.append("--output_directory='{}'".format(paths.MODELS))

    print("Running:")
    print(' '.join(command))
    os.system(' '.join(command))

    print("Export complete: " + datetime.now().isoformat())

