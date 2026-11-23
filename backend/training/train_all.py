"""Train all three models one after another.

Run from the backend folder:
    python -m training.train_all
"""
from training import train_accident, train_rainfall, train_traffic
from training.common import log

if __name__ == "__main__":
    for name, module in [("rainfall", train_rainfall), ("accident", train_accident), ("traffic", train_traffic)]:
        log.info("================ %s ================", name.upper())
        module.main()
    log.info("All models trained. Reports are in backend/reports/.")
