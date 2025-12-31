import io
from datetime import date, timedelta
import pandas as pd
import streamlit as st

# ==============================
# Streamlit config
# ==============================
st.set_page_config(page_title="Interest Expectation Tool", layout="wide")
st.title("Interest Expectation Tool")

# Avoid styler rendering limits by not using .style
pd.set_option("styler.render.max_elements", 1_000_000)

# ==============================
# Session State init
# ==============================
if "uploads_by_day" not in st.session_state:
    # { python_date: {"name": filename, "bytes": b"..."} }
    st.session_state["uploads_by_day"] = {}

if "processed" not in st.session_state:
    st.session_state["processed"] = False
if "summary_base" not in st.session_state:
    st.session_state["summary_base"] = None
if "exposure_df" not in st.session_state:
    st.session_state["exposure_df"] = None
if "edited_df" not in st.session_state:
    st.session_state["edited_df"] = None
if "num_days_in_period" not in st.session_state:
    st.session_state["num_days_in_period"] = None


# =========================================================
# Encoding-safe CSV/TXT reader with delimiter detection
# =========================================================
def read_csv_smart(file_like):
    """
    Reads a .txt (usually comma-separated) or .csv (sometimes colon-separated)
    using common encodings. Auto-detect delimiter from first non-empty line.

    Supports: ',', ':', ';', tab
    """
    encodings = ["utf-8-sig", "cp1256", "windows-1256", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            file_like.seek(0)
            raw = file_like.read()
            text = raw.decode(enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue

        lines = [ln for ln in text.splitlines() if ln.strip()]
        first = lines[0] if lines else ""

        candidates = [",", ":", ";", "\t"]
        best = ","
        best_cols = 1
        for d in candidates:
            cols = first.count(d) + 1
            if cols > best_cols:
                best_cols = cols
                best = d

        buf = io.StringIO(text)
        return pd.read_csv(buf, sep=best, engine="python")

    if last_error is not None:
        raise last_error


def detect_column(columns, candidates):
    """
    Find exact-match column name after stripping outer spaces.
    Returns the original column key from df.columns (preserves exact).
    """
    norm = {str(c).strip(): c for c in columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return None


def normalize_account(x: str) -> str:
    """
    Normalize account strings for exact matching:
    - strip spaces
    - keep as string (do NOT cast to int, to preserve leading zeros)
    """
    if x is None:
        return ""
    return str(x).strip()


# =========================================================
# Lending & commission readers
# =========================================================
def read_single_lending_file(file_like, day_raw):
    """
    Read ONE lending file for a specific day.

    Lending columns (Arabic):
      - name:    'اسم العميل' or 'العميل' or 'اسم'
      - account: 'حساب العميل' (preferred) or 'الكود'
      - debt:    'المديونية'
    """
    try:
        df = read_csv_smart(file_like)
    except Exception as e:
        return None, f"Error reading lending file for {day_raw}: {e}"

    df.columns = [str(c) for c in df.columns]
    cols = list(df.columns)

    name_col = detect_column(cols, ["اسم العميل", "العميل", "اسم"])
    acct_col = detect_column(cols, ["حساب العميل", "الكود"])
    lend_col = detect_column(cols, ["المديونية"])

    missing = []
    if name_col is None:
        missing.append("اسم العميل/العميل/اسم")
    if acct_col is None:
        missing.append("حساب العميل/الكود")
    if lend_col is None:
        missing.append("المديونية")

    if missing:
        return None, (
            f"File for {day_raw} is missing required columns: {', '.join(missing)}\n"
            f"Detected columns: {', '.join(cols)}"
        )

    df = df.rename(columns={name_col: "name", acct_col: "account", lend_col: "lending"})

    df["date"] = pd.to_datetime(day_raw).date()
    df = df[["date", "name", "account", "lending"]].copy()

    df["name"] = df["name"].astype(str).str.strip()
    df["account"] = df["account"].apply(normalize_account)
    df["lending"] = pd.to_numeric(df["lending"], errors="coerce").fillna(0.0)

    # Drop empty accounts (can cause bad matching)
    df = df[df["account"].astype(str).str.strip() != ""].copy()

    return df, None


def read_commission_file(uploaded_file):
    """
    Commission file columns (Arabic, spaces tolerated):
      - account: REQUIRED now (match by account)
          'الكود' or 'حساب العميل'
      - commission: REQUIRED
          'اجمالي العمولات' (or with different hamza/spaces)
      - name: optional ('الاسم' or 'اسم') - not used for matching
    """
    try:
        df = read_csv_smart(uploaded_file)
    except Exception as e:
        return None, f"Error reading commissions file '{getattr(uploaded_file, 'name', '')}': {e}"

    original_cols = [str(c) for c in df.columns]
    stripped_map = {str(c).strip(): c for c in df.columns}

    def find_key(cands):
        for c in cands:
            if c in stripped_map:
                return stripped_map[c]
        return None

    acct_col = find_key(["الكود", "حساب العميل"])
    comm_col = find_key(["اجمالي العمولات", "إجمالي العمولات", "اجمالي العمولات ", " اجمالي العمولات"])

    if acct_col is None or comm_col is None:
        return None, (
            f"Commissions file missing required columns.\n"
            f"Need: الكود/حساب العميل and اجمالي العمولات\n"
            f"Detected columns: {', '.join(original_cols)}"
        )

    out = pd.DataFrame()
    out["account"] = df[acct_col].apply(normalize_account)
    out["commission"] = pd.to_numeric(df[comm_col], errors="coerce").fillna(0.0)

    # Drop empty accounts
    out = out[out["account"].astype(str).str.strip() != ""].copy()

    return out, None


# =========================================================
# Core logic: daily series with special rule
# =========================================================
def build_full_daily_lending(df_lending, start_date, end_date, uploaded_days):
    """
    Build daily series for each (name, account) between start_date and end_date.

    Rule:
    - If a day has an uploaded file:
        missing (name, account) row => lending = 0 that day
    - If a day has NO uploaded file:
        forward-fill from most recent previous day
    """
    if df_lending.empty:
        return pd.DataFrame(columns=["name", "account", "date", "lending"])

    df = df_lending.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    if df.empty:
        return pd.DataFrame(columns=["name", "account", "date", "lending"])

    uploaded_days = sorted({pd.to_datetime(d).normalize() for d in uploaded_days})
    all_dates = pd.date_range(start=start_ts, end=end_ts, freq="D")

    # Universe comes ONLY from uploaded lending files (base truth)
    clients = df[["name", "account"]].drop_duplicates()

    idx = pd.MultiIndex.from_product(
        [clients["name"], clients["account"], all_dates],
        names=["name", "account", "date"],
    )

    df_idx = df.set_index(["name", "account", "date"]).sort_index()[["lending"]]
    df_full = df_idx.reindex(idx)

    # On uploaded days: missing => 0 (NOT ffill)
    is_uploaded_day = df_full.index.get_level_values("date").isin(uploaded_days)
    df_full.loc[is_uploaded_day, "lending"] = df_full.loc[is_uploaded_day, "lending"].fillna(0.0)

    # On non-uploaded days: forward-fill
    df_full["lending"] = (
        df_full.groupby(level=["name", "account"])["lending"]
        .ffill()
        .fillna(0.0)
    )

    df_full = df_full.reset_index()
    df_full["date"] = df_full["date"].dt.date
    return df_full


def compute_summary_by_account(df_daily, df_commissions, default_rate, start_date, end_date):
    """
    One row per (Account Number) from lending files.
    Commission matched by account number EXACTLY.

    Columns:
      - Name (from lending)
      - Account Number (from lending)
      - Rate
      - Total Margin Lending (sum of daily debt over period)
      - Average Margin Lending (Total / number_of_days_in_period)
      - Total Interest (Total Margin Lending * Rate)
      - Total Commission (sum commissions for same account)
      - Difference (max(Total Interest - Total Commission, 0))
    """
    # days in selected period (inclusive)
    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    num_days = max(int(num_days), 1)

    # If a name appears multiple times for same account (rare), take most frequent / last
    # We'll use the latest name observed in df_daily for that account.
    name_by_account = (
        df_daily.sort_values("date")
        .groupby("account")["name"]
        .last()
        .reset_index()
    )

    exposure = (
        df_daily.groupby("account")["lending"]
        .sum()
        .reset_index()
        .rename(columns={"lending": "Total Margin Lending"})
    )

    summary = exposure.merge(name_by_account, on="account", how="left")

    # Commission per account (exact match)
    comm_acc = (
        df_commissions.groupby("account")["commission"]
        .sum()
        .reset_index()
        .rename(columns={"commission": "Total Commission"})
    )

    summary = summary.merge(comm_acc, on="account", how="left")
    summary["Total Commission"] = summary["Total Commission"].fillna(0.0)

    summary = summary.rename(columns={"name": "Name", "account": "Account Number"})

    summary["Rate"] = float(default_rate)
    summary["Average Margin Lending"] = summary["Total Margin Lending"] / num_days
    summary["Total Interest"] = summary["Total Margin Lending"] * summary["Rate"]
    summary["Difference"] = (summary["Total Interest"] - summary["Total Commission"]).clip(lower=0.0)

    # Keep exposure_df for instant recalcs (by Account Number)
    exposure_df = summary[["Account Number", "Total Margin Lending"]].copy()
    return summary, exposure_df, num_days


def recalc_with_new_rates(edited_df, exposure_df, num_days):
    """
    Recalculate Total Interest + Difference after user edits Rate per account.
    """
    merged = edited_df.merge(exposure_df, on="Account Number", how="left", suffixes=("", "_base"))

    # Ensure Total Margin Lending is the base one
    merged["Total Margin Lending"] = merged["Total Margin Lending_base"].fillna(0.0)
    merged = merged.drop(columns=["Total Margin Lending_base"])

    merged["Average Margin Lending"] = merged["Total Margin Lending"] / max(int(num_days), 1)
    merged["Total Interest"] = merged["Total Margin Lending"] * merged["Rate"]
    merged["Difference"] = (merged["Total Interest"] - merged["Total Commission"]).clip(lower=0.0)

    return merged


def create_excel_bytes(df_final):
    cols = [
        "Name",
        "Account Number",
        "Rate",
        "Total Margin Lending",
        "Average Margin Lending",
        "Total Interest",
        "Total Commission",
        "Difference",
    ]
    export = df_final[cols].copy()

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export.to_excel(writer, index=False, sheet_name="Summary")
    buffer.seek(0)
    return buffer


# ==============================
# UI: Date + rate
# ==============================
st.markdown(
    """
**Updated Logic**
- Commission matching is now **by Account Number** (exact match).
- Summary is **one row per Account Number** found in the margin files.
- If a day has a file uploaded and an account is missing → that account’s margin for the day = **0**.
- If a day has no file uploaded → forward-fill using most recent previous uploaded day.
- **Average Margin Lending** = Total Margin Lending ÷ number of days in selected period.
"""
)

col_dates, col_rate = st.columns(2)
with col_dates:
    today = date.today()
    start_date = st.date_input("Start date", value=date(today.year, today.month, 1))
    end_date = st.date_input("End date", value=today)
    if start_date > end_date:
        st.error("Start date cannot be after end date.")

with col_rate:
    default_rate = st.number_input(
        "Default daily interest rate",
        min_value=0.0,
        value=0.0005,
        step=0.0001,
        format="%.6f",
        help="Example: 0.0005 = 0.05% per day",
    )

st.divider()

commission_file = st.file_uploader(
    "Upload commissions file (.txt / .csv)",
    type=["txt", "csv"],
    accept_multiple_files=False,
)

st.divider()

# ==============================
# UI: Calendar grid uploaders (no selection)
# ==============================
st.subheader("Daily lending files (calendar grid)")

uploads_by_day = st.session_state["uploads_by_day"]

all_dates = []
if start_date <= end_date:
    all_dates = list(pd.date_range(start=start_date, end=end_date, freq="D"))

if all_dates:
    first_day = all_dates[0].date()
    first_monday = first_day - timedelta(days=first_day.weekday())
    last_date = all_dates[-1].date()

    st.markdown("Upload a file in the day cell. ✅ means the file is saved (persisted) even if Streamlit reruns.")

    current_date = first_monday
    while current_date <= last_date:
        cols = st.columns(7)
        for i in range(7):
            day = current_date
            with cols[i]:
                if start_date <= day <= end_date:
                    saved = day in uploads_by_day
                    title = f"{day} {'✅' if saved else ''}"
                    st.markdown(f"**{title}**")

                    f = st.file_uploader(
                        "File",
                        type=["txt", "csv"],
                        accept_multiple_files=False,
                        key=f"lend_{day.isoformat()}",
                        label_visibility="collapsed",
                    )

                    # Persist uploaded file to session_state bytes (important for Streamlit Cloud)
                    if f is not None:
                        uploads_by_day[day] = {"name": f.name, "bytes": f.getvalue()}
                        st.session_state["uploads_by_day"] = uploads_by_day
                        st.caption(f"Saved: {f.name}")

                    if saved:
                        if st.button("Remove", key=f"rm_{day.isoformat()}"):
                            uploads_by_day.pop(day, None)
                            st.session_state["uploads_by_day"] = uploads_by_day
                            st.experimental_rerun()
                else:
                    st.write("")
            current_date += timedelta(days=1)

st.divider()

# ==============================
# Processing
# ==============================
process_btn = st.button("Process and calculate", type="primary")

if process_btn:
    if start_date > end_date:
        st.error("Please select a valid date range.")
        st.stop()
    if commission_file is None:
        st.error("Please upload a commissions file.")
        st.stop()

    uploads_by_day = st.session_state.get("uploads_by_day", {})
    if not uploads_by_day:
        st.error("Please upload at least one daily lending file.")
        st.stop()

    # Build lending frames from persisted bytes
    lending_frames = []
    for day, obj in sorted(uploads_by_day.items(), key=lambda x: x[0]):
        bio = io.BytesIO(obj["bytes"])
        bio.name = obj["name"]
        df_day, err = read_single_lending_file(bio, day)
        if err:
            st.error(err)
            st.stop()
        lending_frames.append(df_day)

    df_lending = pd.concat(lending_frames, ignore_index=True)

    df_comm, err = read_commission_file(commission_file)
    if err:
        st.error(err)
        st.stop()

    uploaded_days = df_lending["date"].unique()
    df_daily = build_full_daily_lending(df_lending, start_date, end_date, uploaded_days)

    if df_daily.empty:
        st.warning("No lending data found in the selected date range after processing.")
        st.stop()

    summary_df, exposure_df, num_days = compute_summary_by_account(
        df_daily=df_daily,
        df_commissions=df_comm,
        default_rate=default_rate,
        start_date=start_date,
        end_date=end_date,
    )

    st.session_state["processed"] = True
    st.session_state["summary_base"] = summary_df
    st.session_state["exposure_df"] = exposure_df
    st.session_state["num_days_in_period"] = num_days

    editable_cols = [
        "Name",
        "Account Number",
        "Rate",
        "Total Margin Lending",
        "Average Margin Lending",
        "Total Interest",
        "Total Commission",
        "Difference",
    ]
    st.session_state["edited_df"] = summary_df[editable_cols].copy()

    st.success("Calculation complete.")


# ==============================
# Final screen: Rate editor + final summary + Excel download
# ==============================
if st.session_state["processed"] and st.session_state["edited_df"] is not None:
    st.subheader("Final summary (edit Rate per account)")

    edited_df = st.data_editor(
        st.session_state["edited_df"],
        key="rate_editor",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Rate": st.column_config.NumberColumn(
                "Rate",
                help="Daily interest rate for this account",
                format="%.6f",
                step=0.0001,
                min_value=0.0,
            )
        },
    )
    st.session_state["edited_df"] = edited_df

    updated = recalc_with_new_rates(
        edited_df.copy(),
        st.session_state["exposure_df"],
        st.session_state["num_days_in_period"],
    )

    # Display formatting
    display = updated.copy()
    for c in ["Total Margin Lending", "Average Margin Lending", "Total Interest", "Total Commission", "Difference"]:
        display[c] = pd.to_numeric(display[c], errors="coerce").fillna(0.0).round(2)

    st.dataframe(display, use_container_width=True)

    excel_bytes = create_excel_bytes(updated)
    st.download_button(
        label="Download Excel summary",
        data=excel_bytes,
        file_name=f"interest_summary_{start_date}_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

