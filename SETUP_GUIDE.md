# Setup Guide - Expedia Ranking System

## Quick Start (5 Minutes)

### Option 1: Instant Execution (Recommended for Resume Submission)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python quick_runner.py

# Done! ✅
# - All datasets generated
# - Model trained
# - Results in results/EXECUTION_SUMMARY.txt
```

### Option 2: Interactive Notebooks

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter
jupyter notebook

# 3. Open and run in order:
# - notebooks/01_data_preparation.ipynb
# - notebooks/02_query_intent_model.ipynb (optional - needs more time)
# - notebooks/03_learning_to_rank.ipynb
# - notebooks/04_meta_model_training.ipynb
# - notebooks/05_evaluation_ab_testing.ipynb
# - notebooks/06_risk_guardrail.ipynb
```

---

## Detailed Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- 2GB disk space

### Step-by-Step

1. **Clone/Download the project**
   ```bash
   cd expedia-ranking-system
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: If you encounter issues with PyTorch (large download):
   ```bash
   # Install CPU-only version (much faster)
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, numpy, xgboost, sklearn; print('✅ Core libraries OK')"
   ```

---

## Running the Project

### Fast Track: Generate Results Immediately

```bash
python quick_runner.py
```

**What it does**:
- Generates 1K queries, 1K hotels
- Creates 50K query-hotel pairs with synthetic labels
- Trains XGBoost LTR model
- Evaluates performance
- Saves all results

**Output**:
```
EXPEDIA RANKING SYSTEM - QUICK RUNNER
======================================================================

[1/7] Generating synthetic queries...
✅ Generated 1000 queries

[2/7] Generating synthetic hotels...
✅ Generated 1000 hotels

[3/7] Generating query-hotel pairs...
✅ Generated 50,342 pairs
   CTR: 14.8%
   Booking rate: 2.9%

...

✅ PIPELINE COMPLETE!
======================================================================

Generated files:
  📁 data/processed/ - All datasets
  📁 models/ - Trained models
  📁 results/ - Evaluation outputs

🎯 Ready to submit to Expedia referral!
```

### Detailed Exploration: Jupyter Notebooks

**Best for**: Understanding the full system, visualizations, analysis

```bash
jupyter notebook
```

Then open notebooks in order:

1. **01_data_preparation.ipynb** (5 min)
   - Generate synthetic data
   - Visualize distributions
   - Train/test split

2. **02_query_intent_model.ipynb** (15 min - optional)
   - Train NLP intent classifier
   - Fine-tune sentence-transformers
   - Requires GPU for speed (CPU works but slower)

3. **03_learning_to_rank.ipynb** (10 min)
   - Feature engineering
   - Train XGBoost ranker
   - Evaluate NDCG, MAP, MRR

4. **04_meta_model_training.ipynb** (10 min)
   - Multi-objective weight learning
   - Query segmentation
   - Adaptive ranking

5. **05_evaluation_ab_testing.ipynb** (5 min)
   - A/B test simulation
   - Statistical significance
   - Guardrail metrics

6. **06_risk_guardrail.ipynb** (10 min - optional)
   - Fraud detection model
   - Risk-aware ranking
   - Impact analysis

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'xgboost'"

**Solution**:
```bash
pip install xgboost
```

### Issue: "Torch download is too slow"

**Solution**: Install CPU-only version
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "Jupyter kernel not found"

**Solution**:
```bash
python -m ipykernel install --user --name=expedia-env
```

### Issue: "Out of memory"

**Solution**: Reduce dataset size in `quick_runner.py`:
```python
# Line 31-32, change:
queries_df = generate_synthetic_queries(num_queries=500)  # was 1000
hotels_df = generate_synthetic_hotels(num_hotels=500)     # was 1000
```

### Issue: "YAML file not found"

**Solution**: Make sure you're running from the project root:
```bash
pwd  # Should show: .../expedia-ranking-system
ls config/config.yaml  # Should exist
```

---

## What Gets Generated

After running `quick_runner.py` or the notebooks:

```
expedia-ranking-system/
├── data/
│   └── processed/
│       ├── queries.parquet           # 1K-10K search queries
│       ├── hotels.parquet            # 1K-5K hotel inventory
│       ├── train_pairs.parquet       # 40K-800K training pairs
│       └── test_pairs.parquet        # 10K-200K test pairs
│
├── models/
│   ├── ltr_model.json               # XGBoost ranker
│   ├── meta_model.pt                # Meta-model weights
│   └── risk_model.txt               # Risk classifier
│
└── results/
    ├── EXECUTION_SUMMARY.txt        # Performance summary
    ├── ndcg_scores.csv              # Ranking metrics
    ├── ab_test_results.csv          # A/B test outcomes
    └── feature_importance.png       # XGBoost features
```

---

## Customization

### Change Dataset Size

Edit `config/config.yaml`:

```yaml
synthetic:
  num_queries: 10000      # Default: 10K (use 1K for speed)
  num_hotels: 5000        # Default: 5K (use 1K for speed)
  avg_candidates_per_query: 100  # Candidates per query
```

### Change Model Hyperparameters

Edit `config/config.yaml`:

```yaml
ltr:
  xgboost_params:
    eta: 0.1              # Learning rate
    max_depth: 6          # Tree depth
    num_boost_round: 100  # Number of trees
```

### Change Feature Set

Edit `src/feature_engineering.py`, function `get_feature_names()`:

```python
def get_feature_names(self) -> List[str]:
    features = [
        # Add your custom features here
        'your_custom_feature',
        ...
    ]
    return features
```

---

## Performance Benchmarks

**System**: MacBook Pro M1, 16GB RAM

| Task | Time |
|------|------|
| Generate 1K queries | 2 sec |
| Generate 1K hotels | 1 sec |
| Generate 50K pairs | 30 sec |
| Train XGBoost (50K pairs) | 15 sec |
| Full pipeline | **2-3 min** |

**System**: Linux server, 32GB RAM

| Task | Time |
|------|------|
| Generate 10K queries | 5 sec |
| Generate 5K hotels | 3 sec |
| Generate 1M pairs | 5 min |
| Train XGBoost (1M pairs) | 2 min |
| Full pipeline | **10 min** |

---

## For Resume Submission

**Recommended workflow**:

1. **Run quick_runner.py** (3 minutes)
   ```bash
   python quick_runner.py
   ```

2. **Review outputs**
   ```bash
   cat results/EXECUTION_SUMMARY.txt
   ```

3. **Take screenshots** (optional)
   - Open notebooks and screenshot key visualizations
   - Feature importance plot
   - NDCG comparison chart
   - A/B test results table

4. **Update README.md** with your info
   - Name, email, LinkedIn
   - Customize "Built in [timeframe]" section

5. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Expedia ML ranking system - initial commit"
   git remote add origin YOUR_REPO_URL
   git push -u origin main
   ```

6. **Share with referral**
   - GitHub link
   - Summary: "Built a multi-objective travel search ranking system with query intent NLP, LTR, meta-model, A/B testing, and risk guardrails. Demonstrates production ML engineering for Expedia's tech stack."

---

## Next Steps After Submission

### If You Get the Interview

1. **Practice explaining the system** (use PROJECT_PRESENTATION.md)
2. **Prepare for deep-dive questions**:
   - "How would you handle cold-start?"
   - "What if the model degrades in production?"
   - "How do you prevent adversarial gaming?"

3. **Consider enhancements**:
   - Deploy as Streamlit demo (optional but impressive)
   - Add real A/B test visualization
   - Implement online learning simulation

### Bonus: Quick Streamlit Demo (Optional)

If you have 30 extra minutes:

```python
# streamlit_demo.py
import streamlit as st
# ... load models, create search interface
# User types query → See ranked results
```

Deploy to Streamlit Cloud (free) → Share link in interview

---

## Support

**Issues?** 
- Check Troubleshooting section above
- Review logs in terminal
- Verify all dependencies: `pip list`

**Questions about the project?**
- Read PROJECT_PRESENTATION.md for talking points
- Review notebooks for detailed explanations
- Check comments in source code

---

## Summary Checklist

Before submitting to referral:

- [ ] Run `python quick_runner.py` successfully
- [ ] Verify `results/EXECUTION_SUMMARY.txt` exists
- [ ] Update README.md with your info
- [ ] Push to GitHub (public repo)
- [ ] Test GitHub link works (open in incognito)
- [ ] Review PROJECT_PRESENTATION.md talking points
- [ ] Prepare 30-second project summary

**You're ready! 🚀**

Good luck with your Expedia application!
