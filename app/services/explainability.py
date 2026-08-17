from app.models.manager import model_manager
from src.explainability import explain_prediction_shap

def explain_prediction(X_scaled_record, recommended_servers: int) -> dict:
    """
    Computes local feature contributions for a single prediction and generates
    a clean human-readable explanation sentence using cached explainer.
    """
    explainer = model_manager.shap_explainer
    features = model_manager.capacity_features
    
    if explainer is None or features is None:
        raise RuntimeError("SHAP TreeExplainer assets are not loaded in model manager.")
        
    return explain_prediction_shap(
        explainer=explainer,
        X_scaled_record=X_scaled_record,
        feature_names=features,
        recommended_servers=recommended_servers
    )
