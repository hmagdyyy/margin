import io
from datetime import date, timedelta

import pandas as pd
import streamlit as st

# Allow larger styled tables if needed
pd.set_option("styler.render.max_elements", 1_000_000)

# =========================================================
# Init session state
# =========================================================

if "processed" not in st.session_state:
    st.session_state["processed"] = False
if "summary_df" not in st.session_state:
    st.session_state["summary_df"] = None
if "exposure_df" not in st.session_state:
    st.session_state["exposure_df"] = None
if "edited_df" not in st.session_state:
    st.session_state["edited_df"] = None

# =========================================================
# Encoding-safe CSV/TXT reader with delimiter detection
# =========================================================

def read_csv_smart(uploaded_file):
    """
    Reads a .txt (comma-separated) or .csv (sometimes colon-separated)
    with common Arabic-friendly encodings. Automatically detects the
    delimiter (tries ',', ':', ';', and tab) based on the first non-empty line.
    """
    encodings = ["utf-8-sig", "cp1256", "windows-1256", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            uploaded_file.seek(0)
            raw = uploaded_file.read()
            text = raw.decode(enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue

        # Detect delimiter from first non-empty line
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            # Empty file, just parse as comma-separated
            buffer = io.StringIO(text)
            return pd.read_csv(buffer, sep=",", engine="python")

        first = lines[0]
        candidates = [",", ":", ";", "\t"]
        best_delim = ","
        best_cols = 1
        for d in candidates:
            cols = first.count(d) + 1
            if cols > best_cols:
                best_cols = cols
                best_delim = d

        buffer = io.StringIO(text)
        return pd.read_csv(buffer, sep=best_delim, engine="python")

    if last_error is not None:
        raise last_error

# =========================================================
# Column detection helper
# =========================================================

def detect_column(columns, candidates):
    norm = {str(c).strip(): c for c in columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return None

# =========================================================
# Lending & commission readers
# =========================================================

def read_single_lending_file(uploaded_file, day_raw):
    """
    Read ONE lending file for a specific calendar day.

    Lending file columns (we support multiple variants):
      - name:   'اسم العميل' or 'العميل' or 'اسم'
      - account:'حساب العميل' (primary) or 'الكود'
      - debt:   'المديونية'
    """
    try:
        df = read_csv_smart(uploaded_file)
    except Exception as e:
        return None, f"Error reading lending file '{uploaded_file.name}' for {day_raw}: {e}"

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
            f"File '{uploaded_file.name}' for {day_raw} is missing required columns: "
            + ", ".join(missing)
            + f"\nDetected columns: {', '.join(cols)}"
        )

    # Standardize to internal names
    df = df.rename(
        columns={
            name_col: "name",
            acct_col: "account",
            lend_col: "lending",
        }
    )

    # Force the calendar day into the date column
    day_date = pd.to_datetime(day_raw).date()
    df["date"] = day_date

    df = df[["date", "name", "account", "lending"]].copy()
    df["name"] = df["name"].astype(str).str.strip()
    df["account"] = df["account"].astype(str).str.strip()
    df["lending"] = pd.to_numeric(df["lending"], errors="coerce").fillna(0.0)

    return df, None

def read_commission_file(uploaded_file):
    """
    Commission file columns (we support multiple variants):

      - name:   'الاسم' or 'اسم'
      - account:'الكود' or 'حساب العميل' (optional for calc)
      - total:  'اجمالي العمولات' or 'إجمالي العمولات'
    """
    try:
        df = read_csv_smart(uploaded_file)
    except Exception as e:
        return None, f"Error reading commissions file '{uploaded_file.name}': {e}"

    original_cols = [str(c) for c in df.columns]

    name_col = detect_column(original_cols, ["الاسم", "اسم"])
    acct_col = detect_column(original_cols, ["الكود", "حساب العميل"])
    comm_col = detect_column(original_cols, ["اجمالي العمولات", "إجمالي العمولات"])

    missing = []
    if name_col is None:
        missing.append("الاسم/اسم")
    if comm_col is None:
        missing.append("اجمالي العمولات/إجمالي العمولات")
    if missing:
        return None, (
            f"Commissions file '{uploaded_file.name}' is missing required columns: "
            + ", ".join(missing)
            + f"\nDetected columns: {', '.join(original_cols)}"
        )

    df_norm = pd.DataFrame()
    df_norm["name"] = df[name_col].astype(str).str.strip()

    if acct_col is not None:
        df_norm["account"] = df[acct_col].astype(str).str.strip()
    else:
        df_norm["account"] = ""

    df_norm["commission"] = pd.to_numeric(df[comm_col], errors="coerce").fillna(0.0)
    return df_norm, None

# =========================================================
# Core calculation logic
# =========================================================

def build_full_daily_lending(df_lending, start_date, end_date, uploaded_days):
    """
    Build a daily time series per (name, account) from start_date to end_date.

    Rules:
    - If a file was uploaded for a day:
        * clients absent from that day's file => lending = 0 for that day
        * clients present => use provided value
    - If no file uploaded for a day:
        * forward-fill from most recent previous day
    """
    df = df_lending.copy()
    if df.empty:
        return pd.DataFrame(columns=["name", "account", "date", "lending"])

    # Normalize to midnight timestamps
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    # Restrict to range
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    if df.empty:
        return pd.DataFrame(columns=["name", "account", "date", "lending"])

    # Uploaded days (normalized timestamps)
    uploaded_days = sorted({pd.to_datetime(d).normalize() for d in uploaded_days})

    # Full date range
    all_dates = pd.date_range(start=start_ts, end=end_ts, freq="D")

    # Universe of clients/accounts from ALL uploaded margin files (base truth)
    clients = df[["name", "account"]].drop_duplicates()

    # Build full grid (name, account, date)
    idx = pd.MultiIndex.from_product(
        [clients["name"], clients["account"], all_dates],
        names=["name", "account", "date"],
    )

    # Index the actual snapshots
    df_idx = df.set_index(["name", "account", "date"]).sort_index()[["lending"]]

    # Reindex to full grid (creates NaNs where missing)
    df_full = df_idx.reindex(idx)

    # 1) On uploaded days: missing client => 0 (NOT forward filled)
    # Create a boolean mask for rows that are on uploaded days
    is_uploaded_day = df_full.index.get_level_values("date").isin(uploaded_days)
    df_full.loc[is_uploaded_day, "lending"] = df_full.loc[is_uploaded_day, "lending"].fillna(0.0)

    # 2) On non-uploaded days: forward-fill from last known day
    df_full["lending"] = (
        df_full.groupby(level=["name", "account"])["lending"]
        .ffill()
        .fillna(0.0)
    )

    df_full = df_full.reset_index()
    df_full["date"] = df_full["date"].dt.date
    return df_full


def compute_summary(df_daily, df_commissions, default_rate, df_lending_base):
    """
    One row per client (name):

    - Total Margin Lending = sum of daily lending across all days & accounts
    - Total Interest       = Total Margin Lending * Rate
    - Total Commission     = sum of commissions per client
    - Difference           = max(Total Interest - Total Commission, 0)

    Account Number:
    - Derived ONLY from the daily margin sheets (df_lending_base),
      never from the commissions file.
    - If multiple margin accounts exist for the same name,
      they are joined as comma-separated.
    """

    # 1) Total lending exposure per client (from df_daily)
    exposure = (
        df_daily.groupby("name")["lending"]
        .sum()
        .reset_index()
        .rename(columns={"lending": "total_margin_lending"})
    )

    # 2) Margin accounts per client from BASE lending sheets (only margin accounts)
    def join_accounts(series):
        uniq = sorted({str(a).strip() for a in series if str(a).strip()})
        return ", ".join(uniq)

    margin_accounts = (
        df_lending_base.groupby("name")["account"]
        .apply(join_accounts)
        .reset_index()
        .rename(columns={"account": "Account Number"})
    )

    exposure = exposure.merge(margin_accounts, on="name", how="left")

    # 3) Total commission per client (across all accounts in the commissions file)
    comm_client = (
        df_commissions.groupby("name")["commission"]
        .sum()
        .reset_index()
        .rename(columns={"commission": "total_commission"})
    )

    summary = exposure.merge(comm_client, on="name", how="left")
    summary["total_commission"] = summary["total_commission"].fillna(0.0)

    # 4) Interest + difference
    summary["Rate"] = default_rate
    summary["Total Margin Lending"] = summary["total_margin_lending"]
    summary["Total Interest"] = summary["Total Margin Lending"] * summary["Rate"]
    summary["Difference"] = (
        summary["Total Interest"] - summary["total_commission"]
    ).clip(lower=0.0)

    # 5) Final display columns
    summary = summary.rename(
        columns={
            "name": "Name",
            "total_commission": "Total Commission",
        }
    )

    exposure_df = exposure  # for recalculation (has name + total_margin_lending)
    return summary, exposure_df

def recalc_with_new_rates(summary_df, exposure_df):
    """
    Recalculate Total Interest and Difference after user edits Rate per client.
    """
    exp = exposure_df.rename(columns={"name": "Name"})

    merged = summary_df.merge(
        exp[["Name", "total_margin_lending"]],
        on="Name",
        how="left",
    )

    merged["Total Margin Lending"] = merged["total_margin_lending"]
    merged["Total Interest"] = merged["Total Margin Lending"] * merged["Rate"]
    merged["Difference"] = (
        merged["Total Interest"] - merged["Total Commission"]
    ).clip(lower=0.0)

    merged = merged.drop(columns=["total_margin_lending"])
    return merged

def create_excel_bytes(summary_df):
    """
    Build an in-memory Excel file with the required columns:
    Name, Account Number, Interest Rate, Total Margin Lending,
    Total Interest, Total Commission, Difference
    """
    df_export = summary_df.copy()
    df_export = df_export.rename(columns={"Rate": "Interest Rate"})

    cols = [
        "Name",
        "Account Number",
        "Interest Rate",
        "Total Margin Lending",
        "Total Interest",
        "Total Commission",
        "Difference",
    ]
    df_export = df_export[cols]

    buffer = io.BytesIO()
    import openpyxl  # make sure this is installed: pip install openpyxl
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_export.to_excel(writer, index=False, sheet_name="Summary")
    buffer.seek(0)
    return buffer

# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(page_title="Interest Expectation Tool", layout="wide")
st.title("Interest Expectation Tool (Offline)")

st.markdown(
    "This tool:\n"
    "- Lets you upload daily margin lending files per calendar day\n"
    "- Reads one commissions file for the whole period\n"
    "- Fills missing days using the most recent previous day's lending\n"
    "- Calculates daily interest (debt × rate), summed over the period\n"
    "- Aggregates by client name (across all accounts)\n"
    "- Allows overriding the daily interest rate per client\n"
    "- Exports to Excel with columns: "
    "Name, Account Number, Interest Rate, Total Margin Lending, Total Interest, Total Commission, Difference"
)

st.divider()

# ---- Date range & default rate ----
col_dates, col_rate = st.columns(2)

with col_dates:
    today = date.today()
    start_date = st.date_input(
        "Start date",
        value=date(today.year, today.month, 1),
        help="First date to include in the calculation.",
    )
    end_date = st.date_input(
        "End date",
        value=today,
        help="Last date to include in the calculation.",
    )
    if start_date > end_date:
        st.error("Start date cannot be after end date.")

with col_rate:
    default_rate = st.number_input(
        "Daily interest rate",
        min_value=0.0,
        value=0.0005,
        step=0.0001,
        format="%.6f",
        help="Example: 0.0005 = 0.05% per day",
    )

st.divider()

# ---- Commission file ----
commission_file = st.file_uploader(
    "Upload commissions file (.txt / .csv)",
    type=["txt", "csv"],
    accept_multiple_files=False,
    help="File containing commissions per client over the period.",
)

st.divider()

st.subheader("Daily lending files (calendar-style grid)")

lending_frames = []
all_dates = []

if start_date <= end_date:
    all_dates = list(pd.date_range(start=start_date, end=end_date, freq="D"))

    st.markdown(
        "Upload the margin lending file for each day you have data. "
        "Missing days will automatically use the last available day's values."
    )

    st.markdown("**Calendar** (Mon – Sun)")

    first_day = all_dates[0].date()
    first_monday = first_day - timedelta(days=first_day.weekday())  # Monday=0
    last_date = all_dates[-1].date()

    current_date = first_monday

    while current_date <= last_date:
        cols = st.columns(7)
        for i in range(7):
            day = current_date
            with cols[i]:
                if start_date <= day <= end_date:
                    st.markdown(f"**{day}**")
                    f = st.file_uploader(
                        "File",
                        type=["txt", "csv"],
                        key=f"lend_{day}",
                        label_visibility="collapsed",
                    )
                    if f is not None:
                        df_day, err = read_single_lending_file(f, day)
                        if err:
                            st.error(err)
                            st.stop()
                        lending_frames.append(df_day)
                else:
                    st.markdown("&nbsp;", unsafe_allow_html=True)
            current_date += timedelta(days=1)
else:
    all_dates = []

st.divider()
process_btn = st.button("Process and calculate", type="primary")

if process_btn:
    if len(all_dates) == 0:
        st.error("Please select a valid date range.")
    elif not lending_frames:
        st.error("Please upload at least one daily lending file.")
    elif commission_file is None:
        st.error("Please upload a commissions file.")
    else:
        # ---- Combine lending data ----
        df_lending = pd.concat(lending_frames, ignore_index=True)


        # ---- Read commissions ----
        with st.spinner("Reading commissions file..."):
            df_comm, err = read_commission_file(commission_file)
        if err:
            st.error(err)
            st.stop()

        # ---- Build full daily series ----
        with st.spinner("Building full daily lending (forward-fill)..."):
            uploaded_days = df_lending["date"].unique()
            df_daily = build_full_daily_lending(df_lending, start_date, end_date, uploaded_days)


        if df_daily.empty:
            st.warning("No lending data found in the selected date range after processing.")
            st.stop()

        # ---- Compute summary (initial, using default rate) ----
        with st.spinner("Computing summary..."):
            summary_df, exposure_df = compute_summary(
                df_daily, df_comm, default_rate, df_lending
            )

        # Store in session_state so UI persists on rerun
        st.session_state["summary_df"] = summary_df
        st.session_state["exposure_df"] = exposure_df
        st.session_state["edited_df"] = None
        st.session_state["processed"] = True

        st.success("Calculation complete.")

# =========================================================
# Final summary + editing (only one main screen)
# =========================================================

if st.session_state["processed"] and st.session_state["summary_df"] is not None:
    summary_df = st.session_state["summary_df"]
    exposure_df = st.session_state["exposure_df"]

    st.subheader("Summary – edit daily rate per client")

    editable_cols = [
        "Name",
        "Account Number",
        "Rate",
        "Total Margin Lending",
        "Total Interest",
        "Total Commission",
        "Difference",
    ]

    # Initialize edited_df on first show
    if st.session_state["edited_df"] is None:
        st.session_state["edited_df"] = summary_df[editable_cols].copy()

    edited_df = st.data_editor(
        st.session_state["edited_df"],
        key="summary_editor",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Rate": st.column_config.NumberColumn(
                "Rate",
                help="Daily interest rate for this client",
                format="%.6f",
                step=0.0001,
                min_value=0.0,
            )
        },
    )
    st.session_state["edited_df"] = edited_df

    # Final summary after overrides
    st.subheader("Final summary (after rate overrides)")

    updated_summary = recalc_with_new_rates(edited_df.copy(), exposure_df)
    st.dataframe(updated_summary, use_container_width=True)

    # Excel export of the updated summary
    excel_bytes = create_excel_bytes(updated_summary)
    st.download_button(
        label="Download Excel summary (updated rates)",
        data=excel_bytes,
        file_name=f"interest_summary_{start_date}_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

