from datetime import datetime
import os
import sys
sys.path.append('.')

from src import paths

if __name__ == "__main__":
    try:
        to_training_step = int(sys.argv[1])
    except:
        print("USAGE: pipenv run python scripts/train_model.py <to_training_step>")
        exit(1)

    print("Training beginning: " + datetime.now().isoformat())

    command = []
    command.append("PYTHONPATH=$PYTHONPATH:..:../slim")
    command.append("python " + paths.EXTERNAL_TRAIN_SCRIPT)
    command.append("--pipeline_config_path='{}'".format(paths.MODEL_CONFIG))
    command.append("--model_dir='{}'".format(paths.MODELS))
    command.append("--num_train_steps={}".format(to_training_step))
    command.append("--sample_1_of_n_eval_examples=1")
    command.append("--alsologtostderr")

    print("Running:")
    print(' '.join(command))
    os.system(' '.join(command))

    print("Training complete: " + datetime.now().isoformat())

