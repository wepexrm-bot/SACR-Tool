import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from utils import clean_text, get_stopwords


def explainability_section():
    st.subheader("Explainability — SHAP & LIME")

    # ── Prerequisites ──
    if not st.session_state.get("feature_engineering_done", False):
        st.warning("Please complete Feature Engineering first.")
        return
    if not st.session_state.get("trained_models_all"):
        st.info("No trained models found. Go to the Models tab and click **Train All 5 Models** first.")
        return

    trained = st.session_state.trained_models_all
    vect = st.session_state.vectorizer
    x_test = st.session_state.x_test
    y_test = st.session_state.y_test
    class_names = st.session_state.get("class_names", ["Negative", "Positive"])
    n_classes = len(class_names)

    # ── Pick best model ──
    best_name = max(trained, key=lambda n: trained[n]['f1_weighted'])
    best_clf = trained[best_name]['model']
    best_pipeline = Pipeline(steps=[('vect', vect), ('clf', best_clf)])

    st.success(f"**Best model:** `{best_name}` (F1-weighted = {trained[best_name]['f1_weighted']:.4f})")
    st.caption("SHAP and LIME below use this model. The final tab shows all 5 models side-by-side.")

    tab1, tab2, tab3, tab4 = st.tabs(["SHAP (Global)", "SHAP (Individual)", "LIME", "All 5 Models"])

    # ═══════════════════════════════════════════════════
    #  TAB 1: SHAP Global
    # ═══════════════════════════════════════════════════
    with tab1:
        st.markdown("**SHAP beeswarm summary** — global feature contribution ranking.")

        max_samples = st.slider("SHAP sample size (smaller = faster)", 50, 500, 200, key="shap_samples_global")
        if st.button("Run SHAP (Global)", key="shap_global_btn"):
            with st.spinner("Computing SHAP values (may take a minute)..."):
                try:
                    import shap
                    # Use a small random subset for speed
                    rng = np.random.RandomState(42)
                    idxs = rng.choice(x_test.shape[0], min(max_samples, x_test.shape[0]), replace=False)
                    X_sample = x_test[idxs]

                    # For linear models, use LinearExplainer; otherwise KernelExplainer
                    if hasattr(best_clf, 'coef_'):
                        explainer = shap.LinearExplainer(best_clf, X_sample)
                    else:
                        # Use a background sample for KernelExplainer
                        bg = x_test[rng.choice(x_test.shape[0], min(100, x_test.shape[0]), replace=False)]
                        explainer = shap.KernelExplainer(best_clf.predict_proba, bg)

                    shap_values = explainer.shap_values(X_sample)

                    # Plot beeswarm
                    feature_names = vect.get_feature_names_out()
                    if n_classes == 2:
                        shap.summary_plot(shap_values[1], X_sample.toarray(),
                                          feature_names=feature_names, show=False)
                    else:
                        shap.summary_plot(shap_values, X_sample.toarray(),
                                          feature_names=feature_names, show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()

                    # Waterfall for a single prediction
                    st.markdown("#### Single-instance waterfall")
                    idx = st.number_input("Test-set index for waterfall plot", 0, x_test.shape[0] - 1, 0,
                                          key="shap_waterfall_idx")
                    if n_classes == 2:
                        sv = shap_values[1][idx]
                        exp = shap.Explanation(sv, data=X_sample[idx].toarray()[0],
                                               feature_names=feature_names)
                    else:
                        sv = shap_values[idx]
                        exp = shap.Explanation(sv, data=X_sample[idx].toarray()[0],
                                               feature_names=feature_names)
                    shap.waterfall_plot(exp, show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()

                except ImportError:
                    st.error("`shap` is not installed. Run `pip install shap` and restart.")
                except Exception as e:
                    st.error(f"SHAP error: {e}")

    # ═══════════════════════════════════════════════════
    #  TAB 2: SHAP Individual (force plot for misclassified)
    # ═══════════════════════════════════════════════════
    with tab2:
        st.markdown("**SHAP force plot** on a genuine misclassified example.")
        # Find misclassified
        y_pred_all = best_clf.predict(x_test)
        mis_idx = np.where(y_pred_all != y_test)[0]

        if len(mis_idx) == 0:
            st.info("No misclassified examples found in the test set.")
        else:
            st.write(f"Found {len(mis_idx)} misclassified test samples.")
            mis_choice = st.selectbox(
                "Select misclassified index to explain:",
                mis_idx[:20],
                format_func=lambda i: f"Index {i} — True: {class_names[y_test[i]]}, Pred: {class_names[y_pred_all[i]]}",
                key="shap_mis_idx"
            )
            if st.button("Explain with SHAP", key="shap_mis_btn"):
                with st.spinner("Computing SHAP..."):
                    try:
                        import shap
                        row = x_test[mis_choice]
                        feature_names = vect.get_feature_names_out()
                        if hasattr(best_clf, 'coef_'):
                            explainer = shap.LinearExplainer(best_clf, x_test)
                        else:
                            bg = x_test[np.random.RandomState(42).choice(
                                x_test.shape[0], min(100, x_test.shape[0]), replace=False)]
                            explainer = shap.KernelExplainer(best_clf.predict_proba, bg)
                        shap_val = explainer.shap_values(row)

                        if n_classes == 2:
                            shap.force_plot(explainer.expected_value[1], shap_val[1],
                                            row.toarray()[0], feature_names=feature_names,
                                            matplotlib=True, show=False)
                        else:
                            shap.force_plot(explainer.expected_value[0], shap_val[0],
                                            row.toarray()[0], feature_names=feature_names,
                                            matplotlib=True, show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    except ImportError:
                        st.error("`shap` is not installed.")
                    except Exception as e:
                        st.error(f"SHAP error: {e}")

    # ═══════════════════════════════════════════════════
    #  TAB 3: LIME
    # ═══════════════════════════════════════════════════
    with tab3:
        st.markdown("**LIME** — local explanations for individual predictions.")

        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            st.error("`lime` is not installed. Run `pip install lime` and restart.")
            st.stop()

        # Aliases matching notebook
        def clean_single_text(text):
            return clean_text(text, get_stopwords())

        def explain_with_lime(cleaned_text):
            explainer = LimeTextExplainer(class_names=class_names)
            exp = explainer.explain_instance(
                cleaned_text, best_pipeline.predict_proba, num_features=10, top_labels=n_classes)
            return exp

        def show_lime_explanation(exp):
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)

        st.markdown("##### LIME for a misclassified example")
        y_pred_all_m = best_clf.predict(x_test)
        mis_idx_m = np.where(y_pred_all_m != y_test)[0]
        if len(mis_idx_m) > 0:
            mis_choice_lime = st.selectbox("Pick misclassified index:", mis_idx_m[:20], key="lime_mis_idx",
                                           format_func=lambda i: f"Index {i} — True: {class_names[y_test[i]]}, Pred: {class_names[y_pred_all_m[i]]}")
            if st.button("Explain with LIME", key="lime_mis_btn"):
                cleaned = clean_single_text(st.session_state.x_test_texts[mis_choice_lime])
                exp = explain_with_lime(cleaned)
                for cname in class_names:
                    st.write(f"**P({cname})**: {best_pipeline.predict_proba([cleaned])[0][class_names.index(cname)]:.2%}")
                show_lime_explanation(exp)

        st.markdown("---")
        st.markdown("##### LIME for your own review")
        custom_review = st.text_input("Enter a review to explain with LIME:", "The movie was absolutely fantastic!",
                                      key="lime_custom_input")
        if st.button("Explain with LIME", key="lime_custom_btn"):
            if custom_review.strip():
                cleaned = clean_single_text(custom_review)
                pred = best_pipeline.predict([cleaned])[0]
                st.write(f"**Prediction:** {class_names[pred]}")
                for cname in class_names:
                    st.write(f"P({cname})**: {best_pipeline.predict_proba([cleaned])[0][class_names.index(cname)]:.2%}")
                exp = explain_with_lime(cleaned)
                show_lime_explanation(exp)

    # ═══════════════════════════════════════════════════
    #  TAB 4: All 5 Models on a Custom Review
    # ═══════════════════════════════════════════════════
    with tab4:
        st.markdown("**Compare all 5 models** on your own review.")

        def clean_single_text(text):
            return clean_text(text, get_stopwords())

        user_review = st.text_input("Enter a review to classify:", "This product is amazing and worked perfectly!",
                                    key="all_models_input")
        if st.button("Classify with All Models", key="all_models_btn"):
            if user_review.strip():
                cleaned = clean_single_text(user_review)
                vec = vect.transform([cleaned])

                header = f"{'Model':<25} {'Prediction':<15} {'Confidence':<10}"
                st.code(header)
                st.code("-" * 55)

                for name, entry in trained.items():
                    clf = entry['model']
                    p = clf.predict(vec)[0]
                    lbl = class_names[p]
                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(vec)[0]
                        conf = proba[int(p)]
                        probs_str = " | ".join([f"{c}: {proba[i]:.1%}" for i, c in enumerate(class_names)])
                        st.code(f"{name:<25} {lbl:<15} {conf:.2%}")
                        st.code(f"{'':<25} {probs_str}")
                    else:
                        st.code(f"{name:<25} {lbl:<15} N/A")
                    st.code("-" * 55)
