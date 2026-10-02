
"""
ORACLE - Interactive Annotation Interface
Version: 0.1

Features:
- CIFAR-10 image browsing
- Manual image annotation
- Annotation progress tracking
- Duplicate annotation protection
- CSV export
- Session-based annotation state

Run:
    streamlit run app.py
"""

import random
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import streamlit as st
import torch

from PIL import Image
from torchvision.datasets import CIFAR10


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ORACLE | Active Learning",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CONSTANTS
# ==================================================

DATA_DIR = "data/raw"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

NUM_CLASSES = len(CLASS_NAMES)

INITIAL_SEED = 42


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 750;
            letter-spacing: -1px;
        }

        .subtitle {
            color: #8b8b8b;
            font-size: 1.05rem;
        }

        .metric-card {
            padding: 18px;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
        }

        .section-heading {
            font-size: 1.35rem;
            font-weight: 650;
        }

        div[data-testid="stMetric"] {
            padding: 12px;
            border-radius: 10px;
            border: 1px solid rgba(128,128,128,0.2);
        }

        .stProgress > div > div > div > div {
            background-color: #16a34a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# DATASET LOADING
# ==================================================

@st.cache_resource
def load_dataset():
    """
    Load CIFAR-10 training dataset.

    Labels are retained internally for later evaluation,
    but are never displayed to the annotator.
    """

    dataset = CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
    )

    return dataset


# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================

def initialize_session(dataset_size):
    """Initialize annotation state once per session."""

    if "annotations" not in st.session_state:

        st.session_state.annotations = {}

    if "annotation_history" not in st.session_state:

        st.session_state.annotation_history = []

    if "current_position" not in st.session_state:

        indices = list(
            range(dataset_size)
        )

        random.Random(
            INITIAL_SEED
        ).shuffle(indices)

        st.session_state.sample_order = indices

        st.session_state.current_position = 0

    if "annotation_mode" not in st.session_state:

        st.session_state.annotation_mode = "manual"


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_remaining_indices():
    """Return indices that have not been annotated."""

    return [
        index
        for index in st.session_state.sample_order
        if index not in st.session_state.annotations
    ]


def get_current_index():
    """Return the next available image index."""

    remaining = get_remaining_indices()

    if not remaining:
        return None

    position = min(
        st.session_state.current_position,
        len(remaining) - 1,
    )

    return remaining[position]


def save_annotation(
    image_index,
    selected_label,
):
    """Save a new annotation and update history."""

    if image_index in st.session_state.annotations:

        return False

    annotation = {
        "image_index": int(image_index),
        "label": selected_label,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    st.session_state.annotations[
        image_index
    ] = annotation

    st.session_state.annotation_history.append(
        annotation
    )

    return True


def export_annotations():
    """Convert annotations to a CSV-compatible DataFrame."""

    records = list(
        st.session_state.annotation_history
    )

    return pd.DataFrame(
        records,
        columns=[
            "image_index",
            "label",
            "timestamp",
        ],
    )


# ==================================================
# SIDEBAR
# ==================================================

def render_sidebar(dataset_size):

    st.sidebar.title("🔬 ORACLE")

    st.sidebar.caption(
        "Open Research for Active Learning "
        "and Continuous Evaluation"
    )

    st.sidebar.divider()

    st.sidebar.subheader(
        "Annotation Settings"
    )

    st.sidebar.info(
        "Manual annotation mode: select the "
        "class that best describes each image."
    )

    st.sidebar.metric(
        "Dataset Size",
        f"{dataset_size:,}",
    )

    st.sidebar.metric(
        "Annotated Samples",
        f"{len(st.session_state.annotations):,}",
    )

    st.sidebar.metric(
        "Remaining Samples",
        f"{len(get_remaining_indices()):,}",
    )

    st.sidebar.divider()

    if st.sidebar.button(
        "Reset Session",
        use_container_width=True,
    ):

        for key in [
            "annotations",
            "annotation_history",
            "sample_order",
            "current_position",
            "annotation_mode",
        ]:

            st.session_state.pop(
                key,
                None,
            )

        st.rerun()

    st.sidebar.caption(
        "Resetting removes annotations stored "
        "in the current session."
    )


# ==================================================
# MAIN DASHBOARD
# ==================================================

def render_dashboard(dataset):

    total = len(dataset)

    annotated = len(
        st.session_state.annotations
    )

    remaining = total - annotated

    progress = (
        annotated / total
        if total > 0
        else 0
    )

    st.markdown(
        '<div class="main-title">ORACLE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Interactive Image Annotation Platform'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Images",
        f"{total:,}",
    )

    col2.metric(
        "Annotated",
        f"{annotated:,}",
    )

    col3.metric(
        "Remaining",
        f"{remaining:,}",
    )

    col4.metric(
        "Completion",
        f"{progress * 100:.2f}%",
    )

    st.progress(
        progress,
        text=f"{annotated:,} of {total:,} images annotated",
    )

    st.divider()


# ==================================================
# ANNOTATION INTERFACE
# ==================================================

def render_annotation_interface(dataset):

    st.markdown(
        '<div class="section-heading">'
        'Image Annotation'
        '</div>',
        unsafe_allow_html=True,
    )

    remaining_indices = get_remaining_indices()

    if not remaining_indices:

        st.success(
            "All images have been annotated!"
        )

        st.balloons()

        return

    current_index = get_current_index()

    if current_index is None:
        return

    image, _ = dataset[current_index]

    # ----------------------------------------------
    # Image and controls
    # ----------------------------------------------

    image_col, control_col = st.columns(
        [1.5, 1],
        gap="large",
    )

    with image_col:

        st.caption(
            f"Image ID: {current_index}"
        )

        st.image(
            image,
            caption="Select the correct class",
            width=400,
        )

    with control_col:

        st.subheader(
            "Assign Label"
        )

        st.write(
            "Choose the category that best matches "
            "the displayed image."
        )

        selected_label = st.selectbox(
            "Image Class",
            options=CLASS_NAMES,
            index=None,
            placeholder="Select a class...",
            key=f"label_{current_index}",
        )

        st.write("")

        if st.button(
            "Save Annotation",
            type="primary",
            use_container_width=True,
            disabled=selected_label is None,
        ):

            saved = save_annotation(
                current_index,
                selected_label,
            )

            if saved:

                st.session_state.current_position = 0

                st.success(
                    "Annotation saved successfully!"
                )

                st.rerun()

            else:

                st.warning(
                    "This image has already been annotated."
                )

        st.write("")

        nav_col1, nav_col2 = st.columns(2)

        with nav_col1:

            if st.button(
                "Skip Image",
                use_container_width=True,
            ):

                st.session_state.current_position = (
                    st.session_state.current_position + 1
                ) % len(remaining_indices)

                st.rerun()

        with nav_col2:

            if st.button(
                "Next Image",
                use_container_width=True,
            ):

                st.session_state.current_position = (
                    st.session_state.current_position + 1
                ) % len(remaining_indices)

                st.rerun()


# ==================================================
# ANNOTATION HISTORY
# ==================================================

def render_history():

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        'Annotation History'
        '</div>',
        unsafe_allow_html=True,
    )

    annotations_df = export_annotations()

    if annotations_df.empty:

        st.info(
            "No annotations have been recorded yet."
        )

        return

    st.dataframe(
        annotations_df.tail(20).iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

    # ----------------------------------------------
    # CSV export
    # ----------------------------------------------

    csv_data = annotations_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Annotations (CSV)",
        data=csv_data,
        file_name="oracle_annotations.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ==================================================
# CLASS DISTRIBUTION
# ==================================================

def render_class_distribution():

    annotations_df = export_annotations()

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        'Annotated Class Distribution'
        '</div>',
        unsafe_allow_html=True,
    )

    if annotations_df.empty:

        st.info(
            "Class distribution will appear "
            "after the first annotation."
        )

        return

    counts = (
        annotations_df["label"]
        .value_counts()
        .reindex(
            CLASS_NAMES,
            fill_value=0,
        )
    )

    chart_df = counts.rename(
        "Annotated Samples"
    ).reset_index()

    chart_df.columns = [
        "Class",
        "Annotated Samples",
    ]

    st.bar_chart(
        chart_df,
        x="Class",
        y="Annotated Samples",
    )


# ==================================================
# APPLICATION ENTRY POINT
# ==================================================

def main():

    try:

        dataset = load_dataset()

    except Exception as error:

        st.error(
            "Unable to load CIFAR-10 dataset."
        )

        st.exception(error)

        st.stop()

    initialize_session(
        len(dataset)
    )

    render_sidebar(
        len(dataset)
    )

    render_dashboard(
        dataset
    )

    render_annotation_interface(
        dataset
    )

    render_class_distribution()

    render_history()

    st.divider()

    st.caption(
        "ORACLE v0.1 | Manual Annotation Prototype"
    )


if __name__ == "__main__":

    main()