import streamlit as st
import pandas as pd
import os
import json
import time
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧠 Claims RL Dashboard")

BASE_DIR = "artifacts/experiments"

# -----------------------
# Sidebar Controls
# -----------------------
auto_refresh = st.sidebar.checkbox("🔴 Live Monitoring", value=False)
refresh_rate = st.sidebar.slider("Refresh (sec)", 1, 10, 3)

# -----------------------
# Helpers
# -----------------------
def list_experiments():
    return [
        os.path.join(BASE_DIR, d)
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ]

def load_metrics(exp_path):
    path = os.path.join(exp_path, "metrics.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["experiment"] = os.path.basename(exp_path)
        return df
    return None

def load_config(exp_path):
    path = os.path.join(exp_path, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def load_trajectory(exp_path, episode):
    path = os.path.join(exp_path, "trajectories", f"episode_{episode}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

# -----------------------
# Load experiments
# -----------------------
experiments = list_experiments()

selected = st.sidebar.multiselect(
    "Experiments",
    experiments,
    default=experiments[:1]
)

if not selected:
    st.warning("Select at least one experiment")
    st.stop()

dfs = []
configs = {}

for exp in selected:
    df = load_metrics(exp)
    if df is not None:
        dfs.append(df)
        configs[os.path.basename(exp)] = load_config(exp)

data = pd.concat(dfs)

# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Single Run",
    "📈 Compare",
    "🔍 Episode Drilldown",
    "⚙️ Config"
])

# =======================
# 1. SINGLE RUN + BEST EP
# =======================
with tab1:
    st.header("Single Experiment")

    exp_name = st.selectbox(
        "Experiment",
        [os.path.basename(e) for e in selected]
    )

    exp_path = [e for e in selected if os.path.basename(e) == exp_name][0]
    df = data[data["experiment"] == exp_name]

    df["reward_smooth"] = df["reward"].rolling(10).mean()

    # BEST EPISODE
    best_ep = df.loc[df["reward"].idxmax()]["episode"]

    colA, colB = st.columns(2)

    with colA:
        fig = px.line(df, x="episode", y="reward", title="Reward")
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig = px.line(df, x="episode", y="reward_smooth", title="Smoothed")
        st.plotly_chart(fig, use_container_width=True)

    st.success(f"🏆 Best Episode: {int(best_ep)}")

    # Policy behavior
    st.subheader("Policy Behavior")

    fig = px.line(df, x="episode", y="entropy", title="Entropy (Exploration)")
    st.plotly_chart(fig, use_container_width=True)

    # Action mix
    action_cols = [
        "num_support_actions",
        "num_contradict_actions",
        "num_removed"
    ]

    fig = px.area(df, x="episode", y=action_cols, title="Action Distribution Over Time")
    st.plotly_chart(fig, use_container_width=True)

# =======================
# 2. COMPARE
# =======================
with tab2:
    st.header("Compare Experiments")

    fig = go.Figure()

    for exp in selected:
        name = os.path.basename(exp)
        df = data[data["experiment"] == name]
        df["reward_smooth"] = df["reward"].rolling(10).mean()

        fig.add_trace(go.Scatter(
            x=df["episode"],
            y=df["reward_smooth"],
            mode='lines',
            name=name
        ))

    st.plotly_chart(fig, use_container_width=True)

# =======================
# 3. DRILLDOWN + CLAIM UI
# =======================
with tab3:
    st.header("Episode Drilldown")

    exp_name = st.selectbox(
        "Experiment",
        [os.path.basename(e) for e in selected],
        key="drill"
    )

    exp_path = [e for e in selected if os.path.basename(e) == exp_name][0]
    df = data[data["experiment"] == exp_name]

    ep = st.slider("Episode", int(df["episode"].min()), int(df["episode"].max()))

    traj = load_trajectory(exp_path, ep)

    if traj:
        # CLAIM
        st.subheader("Claim")
        st.write(traj[0].get("claim", "N/A"))

        # EVIDENCE POOL
        st.subheader("Evidence Pool")

        evidence = traj[0].get("evidence_pool", [])

        for e in evidence:
            st.markdown(f"**[{e['id']}]** {e['text']}")

        # REPLAY
        st.subheader("Step Replay")

        step_idx = st.slider("Step", 1, len(traj))

        step = traj[step_idx - 1]

        st.json(step)

        # Highlight selected
        st.subheader("Selected Evidence")

        selected_ids = step.get("selected_ids", [])

        for e in evidence:
            if e["id"] in selected_ids:
                st.success(f"[{e['id']}] {e['text']}")

# =======================
# 4. CONFIG
# =======================
with tab4:
    st.header("Configs")

    for name, cfg in configs.items():
        st.subheader(name)
        st.json(cfg)

# -----------------------
# Live Refresh
# -----------------------
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()