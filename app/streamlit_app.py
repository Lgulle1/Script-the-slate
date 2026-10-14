"""Script the Slate: the app (plan: "Phase 7 app and Phase 6.1 / 6.2"). Read-only over the stored records; the Lines log writes only its own table."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import screens

st.set_page_config(page_title="Script the Slate", layout="wide")
st.title("Script the Slate")

SCREENS = {"Slate": screens.slate, "Player report": screens.player_report, "Matchup": screens.matchup, "Lines log": screens.lines_log, "Grading": screens.grading,
           "Model health": screens.model_health}
if "screen" not in st.session_state:
    st.session_state["screen"] = "Slate"
choice = st.sidebar.radio("Screen", list(SCREENS), key="screen")
SCREENS[choice]()
