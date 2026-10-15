# Deployment Platform

## Platform used: Streamlit Community Cloud
The trained model is deployed as a web application on **Streamlit Community Cloud**, connected directly to this GitHub repository.

| Item | Details |
|---|---|
| Platform | Streamlit Community Cloud (share.streamlit.io) |
| Framework | Streamlit (Python) |
| Source | GitHub repository (branch `main`, main file `app.py`) |
| Python version | 3.12 |
| Cost | Free |
| Update method | Any commit to the GitHub repo redeploys the app automatically |

## Why this platform was chosen
- Free, and needs no server setup or credit card
- Direct GitHub integration, so deployment takes only a few minutes
- Built for Python / machine learning apps, so no separate front-end code is needed
- Gives a public HTTPS link that anyone can open

## Platforms considered
| Platform | Free tier | Ease | Remarks |
|---|---|---|---|
| **Streamlit Community Cloud** | Yes | Very easy | **Chosen**: best fit for an ML demo app |
| Hugging Face Spaces | Yes | Easy | Good alternative, also supports Streamlit / Gradio |
| Render | Yes (sleeps quickly) | Medium | Suited for Flask / FastAPI APIs |
| Heroku / Railway | Mostly paid | Medium | Paid plans needed for always-on |

## Technology stack
- Python 3.12, scikit-learn 1.9.1 (Gradient Boosting model), pandas, numpy, joblib
- Streamlit for the user interface
- GitHub for version control and deployment source

## Limitation
Free Streamlit apps go to sleep after a period of inactivity. A scheduled GitHub Actions workflow (`.github/workflows/keep_alive.yml`) opens the app every 6 hours to keep it awake.
