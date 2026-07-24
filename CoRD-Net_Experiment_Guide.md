# CoRD-Net Experiment Guide

## Prerequisites

-   Activate the virtual environment:

    ``` bash
    source ~/venvs/ml/bin/activate
    ```

-   Install all dependencies.

-   Configure the OAI dataset paths.

-   Verify CUDA availability.

## Experiment Pipeline

  Experiment   Modules Enabled
  ------------ --------------------------
  E1           Baseline ConvNeXt
  E2           STN
  E3           STN + DRP
  E4           STN + Compartment Branch
  E5           E4 + DRP
  E6           E5 + PGR
  E7           E6 + RTC
  E8           E7 + Auxiliary Heads

## Running

``` bash
source ~/venvs/ml/bin/activate
python train.py --config configs/eX.yaml
```

Examples:

``` bash
python train.py --config configs/e1.yaml
python train.py --config configs/e2.yaml
python train.py --config configs/e4.yaml
```

## Workflow

1.  Select the configuration.
2.  Verify enabled modules.
3.  Start training.
4.  Monitor train/validation metrics.
5.  Save checkpoints.
6.  Compare with the previous experiment.

## Metrics

-   Train Accuracy
-   Validation Accuracy
-   Test Accuracy
-   Macro Precision
-   Macro Recall
-   Macro F1
-   QWK
-   MAE
-   Confusion Matrix

## Debugging

-   Visualize STN outputs.
-   Visualize compartment crops.
-   Visualize DRP attention maps.
-   Remove visualization code before long training runs.

## Current Status

-   E1: Complete
-   E2: Complete
-   E3: Pending
-   E4: Cropping fixed; segmentation fault under investigation.
-   E5--E8: Pending

## Notes

-   Keep random seeds fixed.
-   Archive logs, checkpoints, and metrics.
-   Compare each experiment against the previous one before proceeding.
