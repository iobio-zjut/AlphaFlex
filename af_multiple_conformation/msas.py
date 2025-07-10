# Copyright 2021 DeepMind Technologies Limited
"""Full AlphaFold protein structure prediction script."""
import enum
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Any, Dict, Mapping, Union
import copy

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates  # <--- 确保导入 templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax
import jax.numpy as jnp
import numpy as np

# Internal import (7716).

logging.set_verbosity(logging.INFO)


@enum.unique
class ModelsToRelax(enum.Enum):
    ALL = 0
    BEST = 1
    NONE = 2


flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
                         'target that will be folded one after another. If a FASTA file contains '
                         'multiple sequences, then it will be folded as a multimer. Paths should be '
                         'separated by commas. All FASTA paths must have a unique basename as the '
                         'basename is used to name the output directories for each prediction.')

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                                        'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', shutil.which('kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                                                    'database for use by JackHMMER.')
# flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
#                                                   'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                                               'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                                                     'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniref30_database_path', None, 'Path to the UniRef30 '
                                                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                                                   'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                                                 'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the PDB '
                                                      'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                                                'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                                               'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                                                'mapping from obsolete PDB IDs to the PDB IDs of their '
                                                'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset', 'monomer', config.MODEL_PRESETS.keys(),
                  # ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer', 'multimer_v1', 'multimer_v2', 'multimer_v3'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                                         'to obtain a timing that excludes the compilation time, '
                                         'which should be more indicative of the time required for '
                                         'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                                          'pipeline. By default, this is randomly generated. Note '
                                          'that even if this is set, Alphafold may still not be '
                                          'deterministic, because processes like GPU inference are '
                                          'nondeterministic.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 5, 'How many '
                                                              'predictions (each with a different random seed) will be '
                                                              'generated per model. E.g. if this is 2 and there are 5 '
                                                              'models then there will be 10 predictions per input. '
                                                              'Note: this FLAG only applies if model_preset=multimer')
flags.DEFINE_integer('num_monomer_predictions_per_model', 1, 'How many '
                                                             'predictions (each with a different random seed) will be '
                                                             'generated per monomer model. E.g. if this is 2 and there are 5 '
                                                             'models then there will be 10 predictions per input. '
                                                             'Note: this FLAG only applies if model_preset=monomer')
flags.DEFINE_integer('nstruct', 1, 'How many predictions to generate')
flags.DEFINE_integer('nstruct_start', 1, 'model to start with, can be used to parallelize jobs, '
                                         'e.g --nstruct 20 --nstruct_start 20 will only make model _20'
                                         'e.g --nstruct 21 --nstruct_start 20 will make model _20 and _21 etc.')
flags.DEFINE_boolean('use_precomputed_msas', True, 'Whether to read MSAs that '
                                                   'have been written to disk instead of running the MSA '
                                                   'tools. The MSA files are looked up in the output '
                                                   'directory, so it must stay the same between multiple '
                                                   'runs that are to reuse the MSAs. WARNING: This will not '
                                                   'check if the sequence, database or configuration have '
                                                   'changed.')
flags.DEFINE_boolean('seq_only', False, 'Exit after sequence searches')
flags.DEFINE_boolean('no_templates', False, 'Will skip templates faster than using the date')
flags.DEFINE_integer('max_recycles', 3, 'Max recycles')
flags.DEFINE_integer('uniprot_max_hits', 50000, 'Max hits in uniprot MSA')
# flags.DEFINE_integer('mgnify_max_hits', 500, 'Max hits in uniprot MSA')
flags.DEFINE_integer('uniref_max_hits', 10240, 'Max hits in uniprot MSA')
flags.DEFINE_integer('bfd_max_hits', 10240, 'Max hits in uniprot MSA')
flags.DEFINE_float('early_stop_tolerance', 0.5, 'early stopping threshold')
flags.DEFINE_enum_class('models_to_relax', ModelsToRelax.BEST, ModelsToRelax,
                        'The models to run the final relaxation step on. '
                        'If `all`, all models are relaxed, which may be time '
                        'consuming. If `best`, only the most confident model '
                        'is relaxed. If `none`, relaxation is not run. Turning '
                        'off relaxation might result in predictions with '
                        'distracting stereochemical violations but might help '
                        'in case you are having issues with the relaxation '
                        'stage.')
flags.DEFINE_boolean('use_gpu_relax', None, 'Whether to relax on GPU. '
                                            'Relax on GPU can be much faster than CPU, so it is '
                                            'recommended to enable if possible. GPUs must be available'
                                            ' if this setting is enabled.')
flags.DEFINE_boolean('dropout', False, 'Turn on drop out during inference to get more diversity')
flags.DEFINE_boolean('cross_chain_templates', False, 'Whether to include cross-chain distances in multimer templates')
flags.DEFINE_boolean('cross_chain_templates_only', False,
                     'Whether to include cross-chain distances in multimer templates')
flags.DEFINE_boolean('separate_homomer_msas', False, 'Whether to force separate processing of homomer MSAs')
flags.DEFINE_list('models_to_use', None, 'specify which models in model_preset that should be run')
flags.DEFINE_float('msa_rand_fraction', 0, 'Level of MSA randomization (0-1)', lower_bound=0, upper_bound=1)
flags.DEFINE_enum('method', 'afsample2', ['afsample2', 'speachaf', 'af2', 'msasubsampling'],
                  'Choose method from <afsample2, speachaf, af2>')
flags.DEFINE_enum('msa_perturbation_mode', 'random', ['random', 'profile'], 'msa_perturbation_mode')
flags.DEFINE_string('msa_perturbation_profile', None,
                    'A file containing the frequency for the residues that could be randomized')
flags.DEFINE_boolean('use_precomputed_features', False, 'Whether to use precomputed msafeatures')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
    if should_be_set != bool(FLAGS[flag_name].value):
        verb = 'be' if should_be_set else 'not be'
        raise ValueError(f'{flag_name} must {verb} set when running with '
                         f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output


def read_rand_profile():
    msa_frac = {}
    logging.info(f'Reading msa_perturbation_profile from {FLAGS.msa_perturbation_profile}')
    with open(FLAGS.msa_perturbation_profile, 'r') as f:
        for line in f.readlines():
            (pos, frac) = line.rstrip().split()
            msa_frac[int(pos)] = float(frac)
    return msa_frac


def predict_structure(
        fasta_path: str,
        fasta_name: str,
        output_dir_base: str,
        data_pipeline: pipeline.DataPipeline):
    """Only performs MSA search and exits."""
    logging.info('Processing %s for MSA search', fasta_name)
    timings = {}

    msa_output_dir = os.path.join(output_dir_base, 'msas')
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir)

    t_0 = time.time()
    feature_dict = data_pipeline.process(input_fasta_path=fasta_path, msa_output_dir=msa_output_dir)
    timings['msa_search'] = time.time() - t_0

    logging.info('MSA search completed for %s. Time taken: %.1fs', fasta_name, timings['msa_search'])
    logging.info('MSA results saved to %s', msa_output_dir)

    return


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    for tool_name in (
            'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
        if not FLAGS[f'{tool_name}_binary_path'].value:
            raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                             'sure it is installed on your system.')

    use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
    _check_flag('small_bfd_database_path', 'db_preset',
                should_be_set=use_small_bfd)
    _check_flag('bfd_database_path', 'db_preset',
                should_be_set=not use_small_bfd)
    _check_flag('uniref30_database_path', 'db_preset',
                should_be_set=not use_small_bfd)

    run_multimer_system = 'multimer' in FLAGS.model_preset
    _check_flag('pdb70_database_path', 'model_preset',
                should_be_set=not run_multimer_system)
    _check_flag('pdb_seqres_database_path', 'model_preset',
                should_be_set=run_multimer_system)
    _check_flag('uniprot_database_path', 'model_preset',
                should_be_set=run_multimer_system)

    skip_templates = FLAGS.no_templates or FLAGS.seq_only
    monomer_data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        hhblits_binary_path=FLAGS.hhblits_binary_path,
        uniref90_database_path=FLAGS.uniref90_database_path,
        bfd_database_path=FLAGS.bfd_database_path,
        uniref30_database_path=FLAGS.uniref30_database_path,
        small_bfd_database_path=FLAGS.small_bfd_database_path,
        template_searcher=hhsearch.HHSearch(
            binary_path=FLAGS.hhsearch_binary_path,
            databases=[FLAGS.pdb70_database_path]),
        template_featurizer=None if skip_templates else templates.TemplateHitFeaturizer(
            mmcif_dir=FLAGS.template_mmcif_dir,
            max_template_date=FLAGS.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            obsolete_pdbs_path=FLAGS.obsolete_pdbs_path),
        use_small_bfd=use_small_bfd,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
        uniref_max_hits=FLAGS.uniref_max_hits,
        bfd_max_hits=FLAGS.bfd_max_hits,
        no_templates=skip_templates)

    data_pipeline = monomer_data_pipeline

    fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
    for i, fasta_path in enumerate(FLAGS.fasta_paths):
        fasta_name = fasta_names[i]
        output_subdir_for_fasta = os.path.join(FLAGS.output_dir, fasta_name)
        os.makedirs(output_subdir_for_fasta, exist_ok=True)

        predict_structure(
            fasta_path=fasta_path,
            fasta_name=fasta_name,
            output_dir_base=output_subdir_for_fasta,
            data_pipeline=data_pipeline)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'fasta_paths',
        'output_dir',
        'data_dir',
        'uniref90_database_path',
        'bfd_database_path',
        'uniref30_database_path',  
        'template_mmcif_dir',
        'max_template_date',
        'obsolete_pdbs_path',
        'use_gpu_relax',
    ])

    app.run(main)