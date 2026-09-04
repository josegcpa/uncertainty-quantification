set -e

SEEDS=(42 43 44 45 46 123 1234 12345 4242 4243)
EXPERIMENTS=(
    "baseline" 
    "benchmark" 
    "posteriorsampling" 
    "cv_baseline_3"
    "cv_baseline_5"
    "cv_baseline_10" 
    "cv_benchmark_3"
    "cv_benchmark_5" 
    "cv_benchmark_10"
)
CALIBRATIONS=("yes")

run_experiment() {
    seed=$1
    experiment=$2
    calibration=${3:-yes}

    RESULTS_DIR=results
    SAME="--datasets all --virtual-ensembles-count 25 --optimize --n_jobs 1"

    echo "========================================"
    echo "Seed: $seed | Exp: $experiment | Calib: $calibration"
    echo "========================================"

    if [ "$calibration" = "yes" ]
    then
        CALIBRATION="--calibration"
        CALIBRATION_DIR="-calibration"
    else
        CALIBRATION=""
        CALIBRATION_DIR="-nocalibration"
    fi

    # baseline
    if [ "$experiment" == "baseline" ]
    then
        DIR=$RESULTS_DIR/baseline$CALIBRATION_DIR.NA.$seed
        if [ ! -f "$DIR"/.done ]
        then
            uv run script/workflow.py \
                --methods catboost \
                --random-seed $seed \
                --output-dir $DIR $SAME $CALIBRATION && touch $DIR/.done
        fi

    # baseline other methods
    elif [ "$experiment" == "benchmark" ]
    then
        DIR=$RESULTS_DIR/benchmark$CALIBRATION_DIR.NA.$seed
        if [ ! -f "$DIR/.done" ]
        then
            uv run script/workflow.py \
                --methods xgboost lightgbm \
                --random-seed $seed \
                --output-dir $DIR $SAME $CALIBRATION && touch $DIR/.done
        fi

    # posterior sampling
    elif [ "$experiment" == "posteriorsampling" ]
    then
        DIR=$RESULTS_DIR/posteriorsampling$CALIBRATION_DIR.NA.$seed
        if [ ! -f "$DIR/.done" ]
        then
            uv run script/workflow.py \
                --methods catboost \
                --posterior-sampling \
                --random-seed $seed \
                --output-dir $DIR $SAME $CALIBRATION && touch $DIR/.done
        fi

    elif [[ "$experiment" == *"cv_baseline"* ]]
    then
        cv_ensemble_splits=$(echo "$experiment" | cut -d _ -f 3)
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

    elif [[ "$experiment" == *"cv_benchmark"* ]]
    then
        cv_ensemble_splits=$(echo "$experiment" | cut -d _ -f 3)
        DIR=$RESULTS_DIR/benchmarkcvensemble$CALIBRATION_DIR.$cv_ensemble_splits.$seed
        if [ ! -f "$DIR/.done" ]
        then
            uv run script/workflow.py \
                --methods xgboost lightgbm \
                --cv-ensemble \
                --cv-ensemble-splits $cv_ensemble_splits \
                --random-seed $seed \
                --output-dir $DIR $SAME $CALIBRATION && touch $DIR/.done
        fi
    fi 
}
split_data() {
    seed=$1
    uv run script/split_data.py \
        --datasets all \
        --random-seed $seed \
        --test-prop 0.2
}
export -f split_data
export -f run_experiment

parallel split_data ::: "${SEEDS[@]}"
parallel run_experiment ::: "${SEEDS[@]}" ::: "${EXPERIMENTS[@]}" ::: "${CALIBRATIONS[@]}"
