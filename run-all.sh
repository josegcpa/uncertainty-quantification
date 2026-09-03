set -e

N_JOBS=22
SAME="--datasets all --virtual-ensembles-count 25 --optimize --n_jobs $N_JOBS"
RESULTS_DIR=results

for seed in 42 43 44 45 46 123 1234 12345 4242 4243
do
    echo */*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
    echo */*/*/*/*/* Seed: $seed */*/*/*/*/*/
    echo */*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
    for calibration in yes no
    do
        if [ "$calibration" = "yes" ]
        then
            CALIBRATION="--calibration"
            CALIBRATION_DIR="-calibration"
        else
            CALIBRATION=""
            CALIBRATION_DIR="-nocalibration"
        fi

        # baseline
        DIR=$RESULTS_DIR/baseline$CALIBRATION_DIR.NA.$seed
        if [ ! -f "$DIR"/.done ]
        then
            uv run script/workflow.py \
                --methods catboost \
                --random-seed $seed \
                --output-dir $DIR $SAME $CALIBRATION && touch $DIR/.done
        fi

        # baseline other methods
        DIR=$RESULTS_DIR/benchmark$CALIBRATION_DIR.NA.$seed
        if [ ! -f "$DIR/.done" ]
        then
            uv run script/workflow.py \
                --methods xgboost lightgbm \
                --random-seed $seed \
                --output-dir $DIR $SAME $CALIBRATION && touch $DIR/.done
        fi

        # posterior sampling
        DIR=$RESULTS_DIR/posteriorsampling$CALIBRATION_DIR.NA.$seed
        if [ ! -f "$DIR/.done" ]
        then
            uv run script/workflow.py \
                --methods catboost \
                --posterior-sampling \
                --random-seed $seed \
                --output-dir $DIR $SAME $CALIBRATION && touch $DIR/.done
        fi


        for cv_ensemble_splits in 3 5 10
        do
            # CV ensemble
            DIR=$RESULTS_DIR/cvensemble$CALIBRATION_DIR.$cv_ensemble_splits.$seed
            if [ ! -f "$DIR/.done" ]
            then
                uv run script/workflow.py \
                    --methods catboost \
                    --cv-ensemble \
                    --cv-ensemble-splits $cv_ensemble_splits \
                    --random-seed $seed \
                    --output-dir $DIR $SAME $CALIBRATION && touch $DIR/.done
            fi

            # CV ensemble for baselines
            DIR=$RESULTS_DIR/benchmarkcvensemble$CALIBRATION_DIR.$cv_ensemble_splits.$seed
            if [ ! -f "$DIR/.done" ]
            then
                uv run script/workflow.py \
                    --methods xgboost lightgbm rf \
                    --cv-ensemble \
                    --cv-ensemble-splits $cv_ensemble_splits \
                    --random-seed $seed \
                    --output-dir $DIR $SAME $CALIBRATION && touch $DIR/.done
            fi
        done
    done
done
