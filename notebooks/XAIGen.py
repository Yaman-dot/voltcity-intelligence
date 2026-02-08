import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import joblib
import shap
import random
from sklearn.inspection import permutation_importance
import numpy as np

class XAIHandler:
    def __init__(self):
        pass
    def explain_lime(self, x_input_model_space, model, FEATURE_NAMES, explainer, mode="classification"):
        """
        x_input_model_space: 1D numpy array (already encoded + scaled)
        """

        x_input_model_space = np.array(x_input_model_space).flatten()
        # -------------------------
        # Prediction
        # -------------------------
        
        if mode == "classification":
            proba = model.predict_proba(x_input_model_space.reshape(1, -1))[0]
            pred_idx = int(np.argmax(proba))
            pred_class = model.classes_[pred_idx]
            pred_prob = float(proba[pred_idx])
        else:
            pred_class = float(model.predict(x_input_model_space.reshape(1, -1))[0])
            pred_idx = None
            pred_prob = None
        
        # -------------------------
        # LIME explanation
        # -------------------------
        if mode == "classification":
            exp = explainer.explain_instance(
                x_input_model_space,
                model.predict_proba,
                labels=[pred_idx],
                num_features=len(FEATURE_NAMES),
                num_samples=5000
            )
        else:
            exp = explainer.explain_instance(
                x_input_model_space,
                model.predict,
                num_features=len(FEATURE_NAMES),
                num_samples=5000
            )

        records = []
        for rule, value in exp.as_list(label=pred_idx):
            feature_name = self.extract_feature_from_rule(rule, FEATURE_NAMES)

            records.append({
                "feature": feature_name,
                "value": value,
                "abs_value": abs(value)
            })

        if not records:
            return {
                "prediction": pred_class,
                "probability": pred_prob,
                "explanation": ["No feature has a strong influence on this prediction."]
            }


        total_impact = sum(r["abs_value"] for r in records)
        threshold = 0.10 * total_impact

        filtered = [r for r in records if r["abs_value"] >= threshold]


        filtered.sort(key=lambda r: r["abs_value"], reverse=True)


        messages = []
        for r in filtered:
            feature_name = r["feature"].replace("_", " ")
            if r["value"] > 0:
                effect = "increases the likelihood of AI Prediction"
            else:
                effect = "decreases the likelihood of AI Prediction"

            messages.append(f"- {feature_name} {effect}")

        return {
            "prediction": pred_class,
            "probability": pred_prob,
            "explanation": messages
        }
    def explain_shap(self, x_input_model_space, model, FEATURE_NAMES, explainer, mode="classification"):
        """
        x_input_model_space: 1D numpy array (already encoded + scaled)
        """

        x_input_model_space = np.array(x_input_model_space).flatten()
        
        if mode == "classification":
            proba = model.predict_proba(x_input_model_space.reshape(1, -1))[0]
            pred_idx = int(np.argmax(proba))
            pred_class = model.classes_[pred_idx]
            pred_prob = float(proba[pred_idx])
        else:
            pred_class = float(model.predict(x_input_model_space.reshape(1, -1))[0])
            pred_prob = None
            pred_idx = None

        exp = explainer.shap_values(x_input_model_space.reshape(1, -1))
        current_shap_values = exp[0]

        records = []
        for name, value in zip(FEATURE_NAMES, current_shap_values):
            records.append({
                "feature": name,
                "value": value,
                "abs_value": abs(value)
            })

        total_impact = sum(r["abs_value"] for r in records)
        threshold = 0.10 * total_impact

        filtered = [r for r in records if r["abs_value"] >= threshold]


        filtered.sort(key=lambda r: r["abs_value"], reverse=True)


        messages = []
        for r in filtered:
            feature_name = r["feature"].replace("_", " ")
            if r["value"] > 0:
                effect = "increases the likelihood of AI Prediction"
            else:
                effect = "decreases the likelihood of AI Prediction"

            messages.append(f"- {feature_name} {effect}")

        return {
            "prediction": pred_class,
            "probability": pred_prob,
            "explanation": messages
        }
    def extract_feature_from_rule(self, rule, feature_names):
        for fname in feature_names:
            if fname in rule:
                return fname
        return rule
