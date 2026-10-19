
from pathlib import Path

import joblib
import streamlit as st

from preprocessing import clean_text
from nlp_detector import detect_suspicious_phrases
from train_model import train_models


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_model.joblib"
RESULTS_PATH = BASE_DIR / "models" / "model_results.joblib"


st.set_page_config(
    page_title="SMS Shield | Spam Detection",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def get_prediction(model, message):
    prediction = model.predict([message])[0]

    # LinearSVC does not expose predict_proba by default.
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([message])[0]
        spam_index = list(model.classes_).index("spam")
        confidence = float(probabilities[spam_index])

    return prediction, confidence


def risk_color(level):
    return {
        "High": "🔴",
        "Medium": "🟠",
        "Low": "🟢",
    }.get(level, "⚪")


st.title("🛡️ SMS Shield")
st.caption(
    "Intelligent SMS screening with NLP "
    "and Machine Learning"
)

with st.sidebar:
    st.header("About the Project")
    st.write(
        "Detect potential spam messages, identify suspicious "
        "phrases and understand safer next steps."
    )
    st.divider()
    st.write("**Dataset:** SMS Spam Collection")
    st.write("**ML:** Naive Bayes, Logistic Regression, SVM")
    st.write("**NLP:** Phrase and URL pattern detection")
    st.caption(
        "This tool supports awareness. It cannot guarantee "
        "that a message is safe or fraudulent."
    )


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "message_input" not in st.session_state:
    st.session_state.message_input = ""


st.subheader("Analyze a message")

examples = {
    "Choose an example (optional)": "",
    "Possible prize scam": (
        "Congratulations! You have won a cash prize. "
        "Click here to claim now: http://example.com"
    ),
    "Urgent account warning": (
        "URGENT! Your account will be suspended. "
        "Verify your password immediately."
    ),
    "Ordinary message": (
        "Hi, are we still meeting at 5 pm today?"
    ),
}

selected_example = st.selectbox(
    "Try a sample message",
    list(examples.keys()),
)

if st.button("Use selected example", use_container_width=True):
    st.session_state.message_input = examples[selected_example]
    st.session_state.analysis_result = None
    st.rerun()


message = st.text_area(
    "Enter SMS message",
    key="message_input",
    height=150,
    max_chars=5000,
    placeholder="Paste the SMS you want to check...",
)


col1, col2 = st.columns(2)

with col1:
    analyze_clicked = st.button(
        "🔍 Analyze Message",
        type="primary",
        use_container_width=True,
    )

with col2:
    clear_clicked = st.button(
        "🗑️ Clear",
        use_container_width=True,
    )


if clear_clicked:
    st.session_state.analysis_result = None
    st.session_state.message_input = ""
    st.rerun()


if analyze_clicked:
    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        cleaned = clean_text(message)

        with st.spinner("Analyzing message..."):
            nlp_result = detect_suspicious_phrases(cleaned)
            model = load_model()

            if model is None:
                st.error(
                    "The ML model is not trained yet. "
                    "Click Train Models below first."
                )

            else:
                prediction, confidence = get_prediction(
                    model, cleaned
                )

                st.session_state.analysis_result = {
                    "message": message,
                    "prediction": prediction,
                    "confidence": confidence,
                    "nlp": nlp_result,
                }


result = st.session_state.analysis_result


if result:
    st.divider()
    st.subheader("Analysis Results")

    prediction = result["prediction"]
    confidence = result["confidence"]
    nlp = result["nlp"]

    first, second, third = st.columns(3)

    with first:
        if prediction == "spam":
            st.error("🚨 SPAM DETECTED")
        else:
            st.success("✅ HAM — Likely legitimate")

    with second:
        st.metric(
            "Heuristic Risk",
            f'{risk_color(nlp["risk_level"])} '
            f'{nlp["risk_level"]}',
        )

    with third:
        if confidence is not None:
            st.metric(
                "Model Spam Confidence",
                f"{confidence:.1%}",
            )
        else:
            st.metric(
                "Model Confidence",
                "Not available",
            )

    st.caption(
        "Confidence is a model estimate, not a guarantee. "
        "For SVM, probability confidence is not displayed."
    )

    st.subheader("NLP Warning Signs")

    findings = nlp["findings"]

    if findings:
        for category, phrases in findings.items():
            st.markdown(f"**⚠️ {category}**")
            st.write(", ".join(phrases))
    else:
        st.info(
            "No phrases from the current rule list were detected. "
            "This does not prove the message is safe."
        )

    st.subheader("Recommended Safety Steps")

    st.markdown(
        "- Do not share passwords, OTPs, PINs or banking details.\n"
        "- Do not open unexpected or suspicious links.\n"
        "- Verify requests through the organisation's official "
        "app or a trusted contact method.\n"
        "- If you already shared sensitive information, contact "
        "your bank or relevant service through its official channel."
    )


st.divider()
st.subheader("Train and Compare Models")

st.write(
    "Train the three classifiers and compare their test-set "
    "precision, recall, accuracy and F1 score."
)

if st.button("⚙️ Train Models", use_container_width=True):
    try:
        with st.spinner("Training machine learning models..."):
            results_df = train_models()

        load_model.clear()
        st.success("Training completed successfully!")
        st.dataframe(results_df, use_container_width=True)

    except Exception as exc:
        st.error(
            f"Training failed: {type(exc).__name__}: {exc}"
        )


if RESULTS_PATH.exists():
    with st.expander("View saved model comparison"):
        results_df = joblib.load(RESULTS_PATH)
        st.dataframe(results_df, use_container_width=True)
