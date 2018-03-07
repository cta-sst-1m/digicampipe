import os
import pandas as pd
from joblib import Parallel, delayed


def perform_analysis(
    analysis,
    paths,
    output_directory,
    baseline_path='./dark_baseline.npz',
    n_jobs=3,
):
    os.makedirs(output_directory, exist_ok=True)
    kwargs = make_kwargs(analysis, baseline_path, paths, output_directory)

    Parallel(n_jobs=n_jobs)(delayed(part)(**kwarg) for kwarg in kwargs)


def part(analysis, path, baseline_path, outfile_name):
    hillas_parameters = pd.DataFrame(
        analysis(
            files=[path],
            baseline_path=baseline_path,
        )
    )
    run_id = int(path[-11:-8])
    hillas_parameters['run_id'] = run_id
    hillas_parameters.to_json(
        outfile_name,
        lines=True,
        orient='records'
    )
    return True


def make_outfile_name(path, output_directory):
    basename = os.path.split(path)[1]
    basename_no_ext = basename[:-8]
    outfile_name = os.path.join(
        output_directory,
        basename_no_ext + '.jsonl'
    )
    return outfile_name


def make_kwargs(analysis, baseline_path, paths, output_directory):
    kwargs = []
    for path in paths:
        outfile_name = make_outfile_name(path, output_directory)
        if not os.path.isfile(outfile_name):
            kwargs.append({
                'analysis': analysis,
                'path': path,
                'baseline_path': baseline_path,
                'outfile_name': outfile_name,
                })
    return kwargs
