import os

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from cleaning import (
    CLEANED_DATA_PATH,
    COUNTRY_NAMES,
    clean,
    format_country,
    format_distribution_channel,
    format_meal_plan,
    format_room_type,
)
from training.train_model import (
    CAT_FEATURES,
    DATA_PATH,
    MODEL_PATH,
    NUM_FEATURES,
    add_features,
    train,
)

st.set_page_config(
    page_title="Hotel Booking Cancellation Analysis",
    page_icon=None,
    layout="wide",
)

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --page: #0b0d10;
        --surface: #14181e;
        --surface-raised: #1a2028;
        --border: #2a323d;
        --text: #edf2f7;
        --muted: #a4afbd;
        --accent: #55c2d1;
    }
    html, body, [class*="st-"], [data-testid="stAppViewContainer"] {
        color: var(--text);
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: var(--page);
    }
    [data-testid="stHeader"] {
        background: rgba(11, 13, 16, 0.94);
    }
    [data-testid="stSidebar"] {
        background: #101419;
        border-right: 1px solid var(--border);
    }
    h1, h2, h3, h4, p, li, label, [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }
    [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"], small {
        color: var(--muted);
    }
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 16px 18px;
        border-radius: 10px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border-color: var(--border);
        border-radius: 10px;
    }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
    [data-testid="stTextArea"] textarea {
        background: var(--surface-raised);
        color: var(--text);
        border-color: var(--border);
    }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background: var(--surface-raised);
        color: var(--text);
    }
    [data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {
        border-color: var(--border);
    }
    [data-testid="stButton"] button:not(:disabled):hover,
    [data-testid="stFormSubmitButton"] button:not(:disabled):hover {
        border-color: var(--accent);
        color: var(--accent);
    }
    hr { border-color: var(--border); }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="#14181e")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#c6d0dc",
            titleColor="#dce5ee",
            domainColor="#46515e",
            tickColor="#46515e",
            gridColor="#2a323d",
        )
        .configure_legend(labelColor="#c6d0dc", titleColor="#dce5ee")
    )


@st.cache_data(show_spinner="Loading the cleaned hotel booking data...")
def load_bookings() -> pd.DataFrame:
    if os.path.exists(CLEANED_DATA_PATH):
        bookings = pd.read_csv(CLEANED_DATA_PATH)
    elif os.path.exists(DATA_PATH):
        bookings = clean(pd.read_csv(DATA_PATH))
    else:
        raise FileNotFoundError(
            "The cleaned booking data and the original hotel_bookings.csv were not found."
        )

    required_columns = {
        "hotel",
        "is_canceled",
        "arrival_date_year",
        "arrival_date_month",
        "lead_time",
        "market_segment",
    }
    missing_columns = required_columns.difference(bookings.columns)
    if missing_columns:
        raise ValueError(
            "The booking data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    bookings["is_canceled"] = bookings["is_canceled"].astype(int)
    return bookings


def rate_label(value: float) -> str:
    return f"{value:.1%}"


def render_analysis(bookings: pd.DataFrame) -> None:
    st.title("Hotel Booking Cancellation Analysis")
    st.caption(
        "Explore historical booking patterns, compare cancellation rates, "
        "and prepare a clear explanation of the results."
    )

    with st.sidebar:
        st.subheader("Analysis filters")
        years = sorted(bookings["arrival_date_year"].dropna().unique().tolist())
        selected_year = st.selectbox(
            "Arrival year",
            ["All years", *years],
            help="Filter bookings by the year the guest was due to arrive.",
        )
        hotels = ["All properties", *sorted(bookings["hotel"].dropna().unique())]
        selected_hotel = st.selectbox(
            "Hotel",
            hotels,
            help="Compare both properties or focus on one hotel.",
        )

    filtered = bookings
    if selected_year != "All years":
        filtered = filtered[filtered["arrival_date_year"] == selected_year]
    if selected_hotel != "All properties":
        filtered = filtered[filtered["hotel"] == selected_hotel]

    if filtered.empty:
        st.info("No bookings match these filters. Choose a different year or hotel.")
        return

    total_bookings = len(filtered)
    cancellations = int(filtered["is_canceled"].sum())
    cancellation_rate = float(filtered["is_canceled"].mean())
    not_canceled_rate = 1 - cancellation_rate
    average_lead_time = float(filtered["lead_time"].mean())

    st.subheader("At a glance")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Bookings analyzed", f"{total_bookings:,}")
    metric_columns[1].metric("Cancellations", f"{cancellations:,}")
    metric_columns[2].metric("Cancellation rate", rate_label(cancellation_rate))
    metric_columns[3].metric("Average booking lead time", f"{average_lead_time:.0f} days")
    st.caption(
        f"Not-canceled rate: {rate_label(not_canceled_rate)}. "
        "A cancellation rate is the share of bookings marked as canceled."
    )

    st.divider()
    st.subheader("How cancellation rates vary")
    st.caption(
        "Each rate is calculated as canceled bookings divided by all bookings in that group. "
        "Hover over a chart to see the rate and number of bookings."
    )

    month_data = (
        filtered.groupby(["arrival_date_month", "hotel"], observed=True)["is_canceled"]
        .agg(cancellations="sum", bookings="size", cancellation_rate="mean")
        .reset_index()
    )
    month_data["arrival_date_month"] = pd.Categorical(
        month_data["arrival_date_month"], categories=MONTH_ORDER, ordered=True
    )
    month_data = month_data.sort_values("arrival_date_month")

    monthly_chart = (
        alt.Chart(month_data)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X(
                "arrival_date_month:N",
                sort=MONTH_ORDER,
                title="Arrival month",
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y(
                "cancellation_rate:Q",
                title="Cancellation rate",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("hotel:N", title="Hotel"),
            tooltip=[
                alt.Tooltip("hotel:N", title="Hotel"),
                alt.Tooltip("arrival_date_month:N", title="Arrival month"),
                alt.Tooltip("cancellation_rate:Q", title="Cancellation rate", format=".1%"),
                alt.Tooltip("cancellations:Q", title="Canceled bookings", format=","),
                alt.Tooltip("bookings:Q", title="Total bookings", format=","),
            ],
        )
        .properties(height=320)
    )

    market_data = (
        filtered.groupby("market_segment", observed=True)["is_canceled"]
        .agg(cancellations="sum", bookings="size", cancellation_rate="mean")
        .reset_index()
    )
    market_data = market_data[market_data["bookings"] >= 30].sort_values(
        "cancellation_rate", ascending=False
    )
    market_chart = (
        alt.Chart(market_data)
        .mark_bar(color="#367c8b", cornerRadiusEnd=4)
        .encode(
            y=alt.Y(
                "market_segment:N",
                sort="-x",
                title="Booking market segment",
                axis=alt.Axis(labelLimit=180),
            ),
            x=alt.X(
                "cancellation_rate:Q",
                title="Cancellation rate",
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("market_segment:N", title="Market segment"),
                alt.Tooltip("cancellation_rate:Q", title="Cancellation rate", format=".1%"),
                alt.Tooltip("cancellations:Q", title="Canceled bookings", format=","),
                alt.Tooltip("bookings:Q", title="Total bookings", format=","),
            ],
        )
        .properties(height=320)
    )

    chart_columns = st.columns([1.25, 1])
    with chart_columns[0]:
        with st.container(border=True):
            st.markdown("**Cancellation rate by arrival month**")
            st.caption("Compare seasonal patterns across the selected properties.")
            st.altair_chart(style_chart(monthly_chart), use_container_width=True)
    with chart_columns[1]:
        with st.container(border=True):
            st.markdown("**Cancellation rate by market segment**")
            st.caption("Groups with fewer than 30 bookings are hidden to avoid unstable rates.")
            if market_data.empty:
                st.info("There are no market segments with at least 30 bookings in this selection.")
            else:
                st.altair_chart(style_chart(market_chart), use_container_width=True)

    lead_time_bins = [-1, 7, 30, 90, 180, 365, float("inf")]
    lead_time_labels = [
        "0–7 days",
        "8–30 days",
        "31–90 days",
        "91–180 days",
        "181–365 days",
        "366+ days",
    ]
    lead_time_data = filtered.copy()
    lead_time_data["lead_time_group"] = pd.cut(
        lead_time_data["lead_time"],
        bins=lead_time_bins,
        labels=lead_time_labels,
    )
    lead_time_data = (
        lead_time_data.groupby("lead_time_group", observed=False)["is_canceled"]
        .agg(cancellations="sum", bookings="size", cancellation_rate="mean")
        .reset_index()
    )
    lead_chart = (
        alt.Chart(lead_time_data)
        .mark_bar(color="#52749a", cornerRadiusEnd=4)
        .encode(
            x=alt.X("lead_time_group:N", sort=lead_time_labels, title="Days between booking and arrival"),
            y=alt.Y(
                "cancellation_rate:Q",
                title="Cancellation rate",
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("lead_time_group:N", title="Booking lead time"),
                alt.Tooltip("cancellation_rate:Q", title="Cancellation rate", format=".1%"),
                alt.Tooltip("cancellations:Q", title="Canceled bookings", format=","),
                alt.Tooltip("bookings:Q", title="Total bookings", format=","),
            ],
        )
        .properties(height=280)
    )
    with st.container(border=True):
        st.markdown("**Cancellation rate by booking lead time**")
        st.caption(
            "Lead time is the number of days between when a booking was made and the planned arrival."
        )
        st.altair_chart(style_chart(lead_chart), use_container_width=True)

    st.subheader("A simple summary")
    monthly_summary = month_data.dropna(subset=["cancellation_rate"])
    if not monthly_summary.empty:
        highest_month = monthly_summary.loc[monthly_summary["cancellation_rate"].idxmax()]
        st.info(
            f"In the selected data, {highest_month['arrival_date_month']} has the highest "
            f"monthly cancellation rate at {rate_label(float(highest_month['cancellation_rate']))} "
            f"({int(highest_month['bookings']):,} bookings). "
            "This is an observed pattern, not proof that the month causes cancellations."
        )

    with st.expander("How to explain this analysis"):
        st.markdown(
            """
            1. **Start with the question:** What share of hotel bookings were canceled?
            2. **State the result:** Use the cancellation-rate card and mention the number of bookings.
            3. **Compare patterns:** Point to a month, market segment, or booking lead-time group.
            4. **Explain the limit:** These are historical associations. They do not prove that a booking detail caused a cancellation.
            """
        )

    with st.expander("Data and calculation notes"):
        st.markdown(
            """
            - The dashboard uses the project's cleaned Hotel Booking Demand dataset.
            - Cancellation rate = canceled bookings ÷ all bookings in the selected group.
            - The dataset's `is_canceled` label also includes bookings marked as no-shows in the original data.
            - Rates are descriptive and depend on the selected filters and group sample size.
            - The original reservation status fields are excluded because they reveal the booking outcome.
            """
        )


def category_index(values: list[str], preferred: str) -> int:
    return values.index(preferred) if preferred in values else 0


@st.cache_resource(show_spinner="Loading the prediction model...")
def load_model_bundle(model_mtime: float) -> dict:
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        required_models = {"model", "linear_model", "logistic_model", "tree_model"}
        if isinstance(bundle, dict) and required_models.issubset(bundle):
            return bundle
        st.warning("The saved model bundle is outdated. The app will retrain it from the dataset.")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "The prediction model is unavailable and data/hotel_bookings.csv was not found."
        )
    return train()


def render_prediction() -> None:
    st.title("Try a booking prediction")
    st.caption(
        "Enter booking details to see the model's estimated cancellation probability. "
        "This is a prediction, not a guarantee or an explanation of cause."
    )

    model_mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0.0
    bundle = load_model_bundle(model_mtime)

    with st.form("booking_prediction"):
        booking_columns = st.columns(3)
        with booking_columns[0]:
            st.markdown("**Stay details**")
            hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
            month = st.selectbox("Arrival month", MONTH_ORDER)
            lead_time = st.number_input("Days before arrival", 0, 1000, 30)
            weekend_nights = st.number_input("Weekend nights", 0, 30, 1)
            week_nights = st.number_input("Week nights", 0, 60, 2)
            adr = st.number_input("Average daily room rate (EUR)", 0.0, 999.0, 100.0)
        with booking_columns[1]:
            st.markdown("**Guest details**")
            adults = st.number_input("Adults", 0, 10, 2)
            children = st.number_input("Children", 0, 10, 0)
            babies = st.number_input("Babies", 0, 5, 0)
            country_options = sorted(
                set(bundle["countries"]).union(COUNTRY_NAMES).union({"Unknown"})
            )
            country = st.selectbox(
                "Guest country",
                country_options,
                index=category_index(country_options, "GBR"),
                format_func=format_country,
            )
            customer_type = st.selectbox(
                "Customer type",
                bundle["customer_types"],
                index=category_index(bundle["customer_types"], "Transient"),
            )
            repeated_guest = st.checkbox("Returning guest")
        with booking_columns[2]:
            st.markdown("**Booking details**")
            market_segment = st.selectbox(
                "Market segment",
                bundle["market_segments"],
                index=category_index(bundle["market_segments"], "Online TA"),
            )
            distribution_channel = st.selectbox(
                "Distribution channel",
                bundle["distribution_channels"],
                index=category_index(bundle["distribution_channels"], "TA/TO"),
                format_func=format_distribution_channel,
            )
            deposit_type = st.selectbox(
                "Deposit type",
                bundle["deposit_types"],
                index=category_index(bundle["deposit_types"], "No Deposit"),
            )
            meal = st.selectbox(
                "Meal plan", bundle["meals"], format_func=format_meal_plan
            )
            room_type = st.selectbox(
                "Reserved room type",
                bundle["room_types"],
                format_func=format_room_type,
            )
            special_requests = st.number_input("Special requests", 0, 5, 0)

        with st.expander("Additional booking history and preferences"):
            extra_columns = st.columns(3)
            previous_cancellations = extra_columns[0].number_input(
                "Previous cancellations", 0, 30, 0
            )
            previous_completed = extra_columns[1].number_input(
                "Previous completed bookings", 0, 80, 0
            )
            booking_changes = extra_columns[2].number_input("Booking changes", 0, 20, 0)
            parking_spaces = extra_columns[0].number_input("Parking spaces", 0, 8, 0)
            waiting_days = extra_columns[1].number_input("Days on waiting list", 0, 400, 0)

        submitted = st.form_submit_button(
            "Estimate cancellation probability", type="primary", use_container_width=True
        )

    if not submitted:
        return

    total_nights = weekend_nights + week_nights
    total_guests = adults + children + babies
    if total_nights == 0:
        st.warning("Enter at least one night for the stay.")
        return
    if total_guests == 0:
        st.warning("Enter at least one guest.")
        return

    booking = pd.DataFrame(
        [
            {
                "hotel": hotel,
                "arrival_date_month": month,
                "meal": meal,
                "country": country,
                "market_segment": market_segment,
                "distribution_channel": distribution_channel,
                "reserved_room_type": room_type,
                "deposit_type": deposit_type,
                "customer_type": customer_type,
                "lead_time": lead_time,
                "stays_in_weekend_nights": weekend_nights,
                "stays_in_week_nights": week_nights,
                "adults": adults,
                "children": children,
                "babies": babies,
                "is_repeated_guest": int(repeated_guest),
                "previous_cancellations": previous_cancellations,
                "previous_bookings_not_canceled": previous_completed,
                "booking_changes": booking_changes,
                "days_in_waiting_list": waiting_days,
                "adr": adr,
                "required_car_parking_spaces": parking_spaces,
                "total_of_special_requests": special_requests,
            }
        ]
    )
    engineered = add_features(booking)
    features = engineered[NUM_FEATURES + CAT_FEATURES]
    probability = float(bundle["model"].predict_proba(features)[0, 1])
    estimated_value = total_nights * adr

    st.divider()
    st.subheader("Prediction result")
    result_columns = st.columns(3)
    result_columns[0].metric("Estimated cancellation probability", f"{probability:.1%}")
    result_columns[1].metric("Estimated not-canceled probability", f"{1 - probability:.1%}")
    result_columns[2].metric("Estimated stay value", f"€{estimated_value:,.2f}")
    st.progress(probability)
    st.info(
        "The probability comes from the Random Forest model trained on historical bookings. "
        "The training target counts no-shows as canceled. The result is not a certainty and "
        "does not show that any one booking detail caused the outcome."
    )

    importance = bundle.get("importance", {})
    if importance:
        importance_data = pd.DataFrame(
            [
                {"Model feature": name.replace("_", " "), "Relative importance": value}
                for name, value in list(importance.items())[:10]
            ]
        ).sort_values("Relative importance")
        importance_chart = (
            alt.Chart(importance_data)
            .mark_bar(color="#367c8b", cornerRadiusEnd=4)
            .encode(
                x=alt.X("Relative importance:Q", title="Relative importance"),
                y=alt.Y("Model feature:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("Model feature:N"),
                    alt.Tooltip("Relative importance:Q", format=".1%"),
                ],
            )
            .properties(height=280)
        )
        st.markdown("**Features the model relied on most during training**")
        st.caption(
            "Relative importance indicates how useful a feature was to the model. "
            "It does not show whether a feature raises or lowers cancellation risk."
        )
        st.altair_chart(style_chart(importance_chart), use_container_width=True)


try:
    bookings = load_bookings()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.title("Hotel Bookings")
    selected_page = st.radio(
        "Navigate",
        ["Dataset analysis", "Booking prediction"],
        index=0,
    )

if selected_page == "Dataset analysis":
    render_analysis(bookings)
else:
    render_prediction()
