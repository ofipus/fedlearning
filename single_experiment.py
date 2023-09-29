import datetime
import json
import sys
import logging
import argparse

 

from ExperimentUtils import ExperimentUtils

# Initialize logging
logging.basicConfig(level=logging.INFO)

def load_configuration(config_path):
    try:
        with open(config_path) as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)

def validate_configuration(config):
    required_keys = ["model_type", "test_callback", "output_path", "plot_auc", "auc_output_path"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing key in configuration: {key}")
            sys.exit(1)

def run_experiment(config_path):
    config = load_configuration(config_path)
    validate_configuration(config)
    train_data, test_data = ExperimentUtils.simple_train_test_split(config)
    model_class = ExperimentUtils.model_from_config(config['model_type'])
    model = model_class(parameter_config=config)
    results = ExperimentUtils.run_single_experiment(
        model, train_data, test_data, config['test_callback']
    )

    if config['model_type'] not in {'baseline', 'moving_mean_model'}:
        results['lr'] = config['learn_rate']
        ind_results = model.individual_evaluate(test_data)

    if config["plot_auc"]:
        ExperimentUtils.plot_auc(
            results["FPR"],
            results["TPR"],
            results["AUC"],
            str(config["auc_output_path"] + "_(" + config['model_type'] + ")")
        )

    # Removing FPR and TPR for storage reasons
    results.pop("FPR", None)
    results.pop("TPR", None)
    ExperimentUtils.write_to_json(
        ind_results,
        str(config["output_path"] + "_by_user" + "_(" + config['model_type'] + ")")
    )
    ExperimentUtils.write_to_csv(
        results,
        str(config["output_path"] + "_(" + config['model_type'] + ")")
    )
    ExperimentUtils.write_to_json(
        results,
        str(config["output_path"] + "_(" + config['model_type'] + ")")
    )

def main():
    parser = argparse.ArgumentParser(description="Run a single experiment based on provided configuration.")
    parser.add_argument("config_path", help="Path to the JSON configuration file.")
    args = parser.parse_args()
    start = datetime.datetime.now()
    run_experiment(args.config_path)
    finish = datetime.datetime.now() - start
    logging.info(f'Time to finish: {finish.total_seconds()} seconds')
    
if __name__ == '__main__':
    main()
