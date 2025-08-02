import joblib
import json

def load_models_and_stats(asset):
  
    models = {}
    stats = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        # path = f"models/{asset}_model_{col}.pkl"
        model_path = rf"models\{asset}_model_{col}.pkl"
        stats_path = rf"models\{asset}_stats_{col}.json"

        
        models[col] = joblib.load(model_path)

        with open(stats_path, "r") as f:
            stats[col] = json.load(f)  # gives {"mean": ..., "std": ...}

    return models,stats
