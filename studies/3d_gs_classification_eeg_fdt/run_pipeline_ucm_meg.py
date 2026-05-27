import json
import sys
import multiprocessing

from pathlib import Path

from src.data_loading.load_from_table import get_x_arr_for_scikit_from_table
from src.pipelines.train_eval_pipelines import (
    train_eval_gridsearch_loocv_with_outer_n_loop,
)

# Determine the number of CPU cores to use
n_jobs = multiprocessing.cpu_count() - 1  # Leave one core free

# UCM MEG study: 233 subjects in the AD spectrum selected from a large cohort.
# Groups: HC, MCI (non-converter, nC), MCI (converter, C), AD dementia.
# Source reconstruction applied to MEG recordings; generative effective connectivity
# estimated via multivariate Ornstein-Uhlenbeck modelling (Berjaga-Buisan et al. 2025).
# Features: FDT deviation, Asymmetry, and Entropy Production on the GEC matrix.
data_files = {
    "all_features_fdt_gec": "UCM_MEG_gec_analytical_features_ml_ready.csv",
}

classifier = "LogReg"
classifications = [
    "HC_vs_MCI_nC",
    "HC_vs_MCI_C",
    "MCI_nC_vs_MCI_C"
    "HC_vs_AD"
]
group_labels = {"HC": "HC", "MCI_nC": "MCI (nC)", "MCI_C": "MCI (C)", "AD": "AD"}

if __name__ == "__main__":
    # Main execution
    path_repo = Path(Path(__file__).parent / ".." / "..").resolve()
    excel_folder = path_repo / "Data" / "fdt-eeg"
    param_folder = path_repo / "Parameters"

    # Optional CLI arg: index into the classifications list (for SLURM job arrays)
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        run_classifications = [classifications[idx]]
    else:
        run_classifications = classifications

    for classification in run_classifications:
        for data_type in data_files.keys():
            print(f"Training {classifier} with parameters_{classifier}.json")

            results_folder = (
                    path_repo / "Results" / "3d_gs_classification_ucm_meg_fdt_gec" /
                    classifier / classification / data_type
            )
            results_folder.mkdir(exist_ok=True, parents=True)

            data_file = excel_folder / data_files[data_type]

            # Load the parameters file
            param_file = path_repo / "Parameters" / f"parameters_{classifier}_MEG.json"
            with open(param_file, "r") as file:
                parameters = json.load(file)
            parameters["N_JOBS"] = n_jobs

            if classification == "HC_vs_MCI_nC":
                # HC vs MCI non-converter
                parameters["GROUPS"] = {"HC": 0, "MCI (nC)": 1}
            elif classification == "HC_vs_MCI_C":
                # HC vs MCI converter
                parameters["GROUPS"] = {"HC": 0, "MCI (C)": 1}
            elif classification == "MCI_nC_vs_MCI_C":
                # MCI non-converter vs MCI converter: predict AD conversion
                parameters["GROUPS"] = {"MCI (nC)": 0, "MCI (C)": 1}
            else:
                pass

            print(f"Training N fold with {classifier} Grid Search and LOOCV {data_file} Data")

            # We load the data from the file
            x, y, feature_columns = get_x_arr_for_scikit_from_table(
                data_file,
                parameters["GROUPS"],
                parameters["ID_KEY"],
                parameters["GROUP_KEY"]
            )

            # And we run the train_eval_knn gridsearch which performs gridsearch,
            # trains on best parameters, and evaluates the model and reports the
            # performances.
            train_eval_gridsearch_loocv_with_outer_n_loop(
                x, y, feature_columns, parameters, results_folder,
                n=4
            )
