# What was the 99 percent measuring? Auditing an airline loyalty model

Sebastian Leon. A follow-up to my BUSA 480 capstone with American Airlines, which used Kaggle's Airline Passenger Satisfaction survey to "predict Gen Z loyalty" and reported AUC 0.997.

## Reproduced
Same target (`Customer Type == "Loyal Customer"`), same 21 inputs (14 service ratings, distance, delays, age, class, travel type, gender), Gen Z rows (age 27 and under), random 80/20 split, XGBoost as configured: **AUC 0.996, accuracy 98.5%**. Everything below scores on Kaggle's own test file (25,976 passengers).

## Why it is so high
**1. A shortcut.** 99.5 percent of personal travelers are labeled loyal, against 73.7 percent of business travelers. Loyalty rises with age (62.6 percent for Gen Z, 95.7 percent for Boomers). Age, travel type and class alone, with no service ratings, reach AUC 0.90 (0.92 for Gen Z).

| Inputs | AUC, all | AUC, Gen Z |
|---|---|---|
| Everyone loyal (majority guess) | 0.500 (81.5% accurate) | 0.500 (60.8% accurate) |
| Age | 0.793 | 0.673 |
| Age + travel type | 0.881 | 0.845 |
| Age + travel type + class | 0.901 | 0.922 |
| 14 service ratings only | 0.951 | 0.885 |
| All capstone inputs | 0.997 | 0.997 |

**2. A data artifact.** Among business travelers, the 14 service ratings alone separate loyal from disloyal at AUC 0.998 on the holdout. Real survey answers rarely separate a group that cleanly; it takes combinations of ratings (a depth-3 tree gets 0.87, boosting 0.998). One visible sign: business travelers who answered 0, "not applicable", for departure-time convenience are 74 percent disloyal, against 26 percent overall. This is consistent with simulated or heavily processed data. I could not verify the dataset's origin.

## What that means
The model was accurate, but it was learning how the dataset was built, not what makes a traveler loyal. "Customer Type" is also not loyalty-program enrollment. Answering the business question needs American's own data (AAdvantage enrollment and behavior), a holdout split, and a design that separates service experience from who travels for work.

## Run it
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) into `data/raw/` (train.csv, test.csv), then:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
brew install libomp   # macOS, for XGBoost and LightGBM
.venv/bin/python src/loyalty.py
```
