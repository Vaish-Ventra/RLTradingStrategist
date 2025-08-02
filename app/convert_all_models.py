import joblib
import importlib

assets = ["btc", "eth"]
components = ["open", "high", "low", "close", "volume"]

for asset in assets:
    for comp in components:
        try:
            # Import the model module: models.btc.close
            module_path = f"models.{asset}.{comp}"
            mod = importlib.import_module(module_path)

            # Get the 'model' object defined inside that module
            model = getattr(mod, "model", None)
            if model is None:
                print(f"❌ No 'model' variable in {module_path}")
                continue

            # Save as .pkl
            output_path = f"models/{asset}_model_{comp}.pkl"
            joblib.dump(model, output_path)
            print(f"✅ Saved: {output_path}")
        except Exception as e:
            print(f"❌ Failed to convert {asset} {comp}: {e}")
