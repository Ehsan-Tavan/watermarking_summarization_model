# ========================== Third Party libs ========================
import os
# ============================ My packages ============================
from src.data_reader import read_json
from src.configurations import BaseConfig
from src.watermark_analyzer import CoreLib

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    data = read_json(os.path.join(ARGS.assets_dir, ARGS.evaluation_output_file))

    CORE = CoreLib(data, simcse_model_path=ARGS.simcse_model_path, device=ARGS.device)

    metrics2value = CORE.run()
    print(metrics2value)
