# Hotel Booking Cancellation Analysis

A Streamlit dashboard for exploring hotel booking patterns and cancellation
rates, with a trained model for cancellation analysis.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with
   GitHub and connect the `khouen-menghor/demo_data_analysis_hotel_booking`
   repository.
2. Select `main` as the branch and `app.py` as the app file.
3. In advanced settings, select Python 3.12 to match the saved model runtime.
4. Deploy the app.

The dependency versions in `requirements.txt` match the environment used to
create the checked-in model.
