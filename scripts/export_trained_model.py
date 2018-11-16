from datetime import datetime
import os
import subprocess
import re
import sys
sys.path.append('.')

from src import paths

if __name__ == "__main__":
    print("Export beginning: " + datetime.now().isoformat())

    model_regex = re.compile(r"model.ckpt-([0-9]+)")
    matches = [model_regex.match(f) for f in os.listdir(paths.MODELS)]
    checkpoints = [m for m in matches if m is not None]

    sorter = lambda c: -int(c.group(1))
    last_checkpoint = sorted(checkpoints, key=sorter)[0]

    command = []
    command.append("pipenv run python " + paths.EXTERNAL_EXPORT_SCRIPT)
    command.append("--input_type='image_tensor'")
    command.append("--pipeline_config_path='{}'".format(paths.MODEL_CONFIG))
    command.append("--trained_checkpoint_prefix='{}'".format(os.path.join(paths.MODELS, last_checkpoint.group(0))))
    command.append("--output_directory='{}'".format(paths.MODELS))

    print("Running:")
    print(' '.join(command))
    subprocess.run(command, shell=True, check=True)

    print("Export complete: " + datetime.now().isoformat())

