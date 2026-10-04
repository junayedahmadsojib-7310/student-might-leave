import numpy as np
import pandas as pd
import streamlit as st

import portable

URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"

st.set_page_config(page_title="Study office: who to talk to", layout="wide")
st.title("Study office: who should we talk to in week 6?")


@st.cache_data
def load_data():
    history = pd.read_csv(URL + "history_week6.csv")
    new = pd.read_csv(URL + "new_week6.csv")
    return history, new


@st.cache_resource
def load_model():
    return portable.Model("model")


history, new = load_data()
model = load_model()

# risk for this year's students, and for the 2025 cohort (where we know what happened)
new["risk"] = model.predict_proba(new)
val = history[history["cohort"] == 2025].copy()
val["risk"] = model.predict_proba(val)


def four_boxes(frame, flagged):
    """Count the four boxes for a group of students and a True/False 'contacted' column."""
    left = frame["left"] == 1
    flagged = flagged.astype(bool)
    tp = int((left & flagged).sum())
    fp = int((~left & flagged).sum())
    fn = int((left & ~flagged).sum())
    tn = int((~left & ~flagged).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return tn, fp, fn, tp, precision, recall


def in_words(tn, fp, fn, tp):
    return (f"**{tp}** students reached in time, **{fp}** worried for nothing, "
            f"**{fn}** missed, **{tn}** correctly left alone.")


# ---------- sidebar: the rule ----------
st.sidebar.header("The rule")
mode = st.sidebar.radio("Choose the rule by", ["Number of conversations", "Risk cut-off"])
if mode == "Number of conversations":
    k = st.sidebar.slider("Conversations", 10, 200, 40)
    val["contacted"] = val["risk"].rank(ascending=False, method="first") <= k
    new["contacted"] = new["risk"].rank(ascending=False, method="first") <= k
    st.sidebar.caption(f"Talk to the {k} students with the highest risk.")
else:
    cut = st.sidebar.slider("Cut-off", 0.02, 0.90, 0.20, 0.01)
    val["contacted"] = val["risk"] >= cut
    new["contacted"] = new["risk"] >= cut
    st.sidebar.caption(f"Talk to everyone with a risk of at least {cut:.0%}.")

tab1, tab2, tab3, tab4 = st.tabs(["This week's list", "Mistakes of the rule",
                                  "Per group", "What does it cost?"])

# ---------- 1. this week's list ----------
with tab1:
    st.subheader("2026 students, highest risk first")
    st.write(f"**{int(new['contacted'].sum())}** students are marked for a conversation.")
    show = new.sort_values("risk", ascending=False)[
        ["student_id", "contacted", "risk", "programme", "international", "fees_owed",
         "submitted_share", "missed_last3", "logins_trend", "quiz_mean"]]
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={"risk": st.column_config.NumberColumn("risk", format="%.2f")})

# ---------- 2. mistakes ----------
with tab2:
    st.subheader("What would this rule have done in 2025?")
    tn, fp, fn, tp, precision, recall = four_boxes(val, val["contacted"])
    st.write(in_words(tn, fp, fn, tp))
    c1, c2 = st.columns(2)
    c1.metric("Precision", f"{precision:.0%}",
              help="Of the students we contacted, the share who were really at risk")
    c2.metric("Recall", f"{recall:.0%}",
              help="Of the students who left, the share we reached")
    st.table(pd.DataFrame({"not contacted": [f"{tn} stayed (fine)", f"{fn} left (missed)"],
                           "contacted": [f"{fp} stayed (false alarm)", f"{tp} left (reached)"]},
                          index=["stayed", "left"]))

# ---------- 3. per group ----------
with tab3:
    st.subheader("International and domestic students")
    rows = []
    for flag, name in [(1, "International"), (0, "Domestic")]:
        g = val[val["international"] == flag]
        tn, fp, fn, tp, precision, recall = four_boxes(g, g["contacted"])
        rows.append({"group": name, "students": len(g), "really left": int(g["left"].sum()),
                     "reached": tp, "worried for nothing": fp, "missed": fn,
                     "precision": f"{precision:.0%}", "recall": f"{recall:.0%}",
                     "average predicted risk": round(g["risk"].mean(), 3),
                     "share who really left": round(g["left"].mean(), 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    logins = history.groupby("international")["logins_total"].mean()
    st.write(f"Average logins in weeks 1-6: domestic **{logins[0]:.0f}**, "
             f"international **{logins[1]:.0f}**. If a login means something different "
             "for the two groups, the model may treat them unequally.")

# ---------- 4. my own: costs ----------
with tab4:
    st.subheader("Set the costs, find the best cut-off")
    a, b, c, d = st.columns(4)
    talk = a.number_input("A conversation (DKK)", value=500, step=100)
    worry = b.number_input("A false alarm (DKK)", value=2000, step=500)
    leave = c.number_input("A student who leaves (DKK)", value=60000, step=5000)
    helps = d.slider("Conversation keeps (share)", 0.05, 1.0, 0.30, 0.05)

    cuts = np.round(np.arange(0.02, 0.91, 0.01), 2)
    values = []
    for cu in cuts:
        flagged = val["risk"] >= cu
        tp_ = int(((val["left"] == 1) & flagged).sum())
        fp_ = int(((val["left"] == 0) & flagged).sum())
        values.append(tp_ * helps * leave - (tp_ + fp_) * talk - fp_ * worry)
    curve = pd.Series(values, index=cuts, name="net value (DKK)")
    best_cut = float(curve.idxmax())
    st.line_chart(curve)
    n_best = int((new["risk"] >= best_cut).sum())
    st.write(f"Best cut-off on 2025: **{best_cut:.2f}**, net value **{curve.max():,.0f} DKK**. "
             f"This year that means **{n_best}** conversations. The office can hold about 40.")
