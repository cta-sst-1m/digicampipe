import os
from tqdm import tqdm
import pandas as pd


def perform_analysis(
    analysis,
    paths,
    output_directory,
    baseline_path='./dark_baseline.npz',
):
    os.makedirs(output_directory, exist_ok=True)
    for path in tqdm(paths):
        basename = os.path.split(path)[1]
        basename_no_ext = basename[:-8]
        run_id = int(basename_no_ext[-3:])
        outfile_name = os.path.join(
            output_directory,
            basename_no_ext + '.jsonl'
        )
        if not os.path.isfile(outfile_name):
            hillas_parameters = pd.DataFrame(
                analysis(
                    files=[path],
                    baseline_path=baseline_path,
                )
            )
            hillas_parameters['run_id'] = run_id
            hillas_parameters.to_json(
                outfile_name,
                lines=True,
                orient='records'
            )
