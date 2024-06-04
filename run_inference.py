import sys
from src.models.model_T5 import FlanT5FineTuner
from src.utils.config import CONFIG

def main(mode):
    model_name = CONFIG["model_name"]
    model_dir = CONFIG["models_dir"]

    model = FlanT5FineTuner(model_name, model_dir)

    if mode == "zero_shot":
        model.run_zero_shot_inference()
    elif mode == "one_shot":
        model.run_one_shot_inference()
    else:
        print(f"Unknown mode: {mode}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/run_inference.py [zero_shot|one_shot]")
        sys.exit(1)

    mode = sys.argv[1]
    main(mode)
