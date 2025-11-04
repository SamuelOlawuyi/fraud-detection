# Demo Video Script (3 Minutes)

## Recording Setup

**Tools**:
- Screen recorder: OBS Studio, Loom, or QuickTime
- Resolution: 1920x1080 (1080p)
- Audio: Clear microphone, minimal background noise
- Duration: 3 minutes (180 seconds)

**What to Show**:
1. Project overview and architecture
2. Live API demonstration
3. Model performance metrics
4. Deployment and CI/CD

---

## Script

### [0:00 - 0:30] Introduction (30 seconds)

**Visual**: GitHub repository homepage

**Script**:
> "Hi! I'm [Your Name], and today I'll demonstrate my end-to-end fraud detection system. This project showcases a complete machine learning pipeline from data ingestion to production deployment.
>
> The system uses Random Forest with SMOTE to detect fraudulent transactions in real-time, achieving 95% precision and 87% recall. It's built with Python, FastAPI, Docker, and deployed on Render with full CI/CD."

**Actions**:
- Show README with project badges
- Scroll through project structure
- Highlight key features

---

### [0:30 - 1:15] System Architecture & Code Walkthrough (45 seconds)

**Visual**: VS Code with project open

**Script**:
> "Let's look at the architecture. The system has four main components:
>
> 1. **Data Pipeline**: Downloads and preprocesses the Kaggle credit card dataset
> 2. **Feature Engineering**: Creates 20+ features including temporal patterns and statistical aggregations
> 3. **Model Training**: Random Forest with SMOTE handles the severe class imbalance
> 4. **REST API**: FastAPI serves real-time predictions
>
> The code is modular, well-tested, and follows best practices."

**Actions**:
- Show `src/` directory structure
- Open `src/utils/preprocess.py` - highlight feature engineering
- Open `src/models/train.py` - show SMOTE integration
- Open `src/api/main.py` - show API endpoints

---

### [1:15 - 2:00] Live API Demonstration (45 seconds)

**Visual**: Browser with API documentation (Swagger UI)

**Script**:
> "Now let's see it in action. Here's the live API deployed on Render.
>
> The `/predict` endpoint accepts transaction data and returns a fraud risk score. Let me send a sample transaction...
>
> The API responds in under 50 milliseconds with a fraud probability of 0.87 and risk level 'high'. The system also supports batch predictions for high-throughput scenarios.
>
> The `/health` endpoint shows the model is loaded and ready, and `/model/info` provides metadata about the deployed model."

**Actions**:
- Navigate to `https://your-app.onrender.com/docs`
- Expand `/predict` endpoint
- Click "Try it out"
- Paste example transaction JSON
- Execute and show response
- Show `/health` and `/model/info` endpoints

**Example Transaction**:
```json
{
  "Time": 12345.0,
  "Amount": 100.50,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.551600,
  "V12": -0.617801,
  "V13": -0.991390,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053
}
```

---

### [2:00 - 2:30] Model Performance & Evaluation (30 seconds)

**Visual**: Jupyter notebook or evaluation plots

**Script**:
> "The model's performance is impressive. Looking at the evaluation metrics:
>
> - 99.8% accuracy
> - 95.2% precision - meaning only 5% false positives
> - 87.3% recall - catching 87% of all fraud
> - AUC-ROC of 0.98
>
> The confusion matrix shows we correctly classify 99.98% of legitimate transactions while catching most fraud cases. This balance is crucial for real-world deployment."

**Actions**:
- Show confusion matrix plot
- Show ROC curve
- Show precision-recall curve
- Highlight key metrics

---

### [2:30 - 3:00] Deployment & CI/CD (30 seconds)

**Visual**: GitHub Actions and Render dashboard

**Script**:
> "The system is production-ready with full CI/CD. Every push triggers automated tests via GitHub Actions. The Docker container is built and deployed automatically to Render.
>
> The repository includes comprehensive documentation, unit tests with 90%+ coverage, and a detailed case study.
>
> This project demonstrates end-to-end ML engineering: from data science to production deployment. Thank you for watching! Check out the repository for more details."

**Actions**:
- Show GitHub Actions workflow (green checkmarks)
- Show Render deployment logs
- Show test coverage report
- End on GitHub repo with star button visible

---

## Recording Tips

### Before Recording
1. **Prepare environment**:
   - Close unnecessary tabs/applications
   - Clear browser history for clean demo
   - Test API endpoints beforehand
   - Have example transactions ready to copy-paste

2. **Practice**:
   - Rehearse the script 2-3 times
   - Time each section
   - Ensure smooth transitions

3. **Technical setup**:
   - Ensure API is deployed and running
   - Have all plots/visualizations ready
   - Test screen recording software

### During Recording
1. **Speak clearly** and at moderate pace
2. **Pause briefly** between sections
3. **Show, don't just tell** - let visuals support your words
4. **Keep mouse movements smooth**
5. **If you make a mistake**, pause and restart that section

### After Recording
1. **Edit** for clarity (remove long pauses, mistakes)
2. **Add captions** for accessibility
3. **Add intro/outro** with your name and links
4. **Export** in 1080p, MP4 format
5. **Upload** to YouTube/Vimeo
6. **Update README** with video link

---

## Alternative: Shorter Demo (1 Minute)

If you need a shorter version:

**[0:00-0:15]** Quick intro + architecture overview
**[0:15-0:40]** Live API demo with one prediction
**[0:40-0:55]** Show key metrics and plots
**[0:55-1:00]** Closing with repository link

---

## Video Hosting Options

1. **YouTube** (Recommended)
   - Unlisted or public
   - Good for sharing with recruiters
   - Add to LinkedIn

2. **Loom**
   - Quick and easy
   - Good for personal sharing
   - Limited analytics

3. **Vimeo**
   - Professional appearance
   - Better privacy controls
   - May require paid plan

---

## Checklist

- [ ] API is deployed and accessible
- [ ] All plots and visualizations are generated
- [ ] Example transactions are prepared
- [ ] Script is practiced and timed
- [ ] Recording software is tested
- [ ] Audio quality is verified
- [ ] Screen resolution is 1080p
- [ ] Video is edited and polished
- [ ] Captions are added
- [ ] Video is uploaded
- [ ] Link is added to README
- [ ] Link is shared on LinkedIn/portfolio

---

**Good luck with your demo! 🎥**