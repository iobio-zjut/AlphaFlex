import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import msa_identifiers
from alphafold.data import parsers
from alphafold.data import templates # This import is crucial for templates.TemplateHitFeaturizer
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.data.tools import jackhmmer
import numpy as np

# We'll use the numerical values for RESTYPE_NUM and ATOM_TYPE_NUM directly,
# as they don't seem to be exposed as attributes in your residue_constants module.
# Standard AlphaFold values:
STANDARD_RESTYPE_NUM = 21 # 20 standard amino acids + 1 for unknown (X)
STANDARD_ATOM_TYPE_NUM = 37 # Max atoms per residue in PDB

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


def make_sequence_features(
        sequence: str, description: str, num_res: int) -> FeatureDict:
    """Constructs a feature dict of sequence features."""
    features = {}
    features['aatype'] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True)
    features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
    features['domain_name'] = np.array([description.encode('utf-8')],
                                       dtype=np.object_)
    features['residue_index'] = np.array(range(num_res), dtype=np.int32)
    features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
    features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
    return features


def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError('At least one MSA must be provided.')

    int_msa = []
    deletion_matrix = []
    species_ids = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
        for sequence_index, sequence in enumerate(msa.sequences):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
            deletion_matrix.append(msa.deletion_matrix[sequence_index])
            identifiers = msa_identifiers.get_identifiers(
                msa.descriptions[sequence_index])
            species_ids.append(identifiers.species_id.encode('utf-8'))

    num_res = len(msas[0].sequences[0])
    num_alignments = len(int_msa)
    features = {}
    features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
    features['msa'] = np.array(int_msa, dtype=np.int32)
    features['num_alignments'] = np.array(
        [num_alignments] * num_res, dtype=np.int32)
    features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
    return features


def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
    """Runs an MSA tool, checking if output already exists first."""
    if not use_precomputed_msas or not os.path.exists(msa_out_path):
        if max_sto_sequences is not None:
            result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]  # pytype: disable=wrong-arg-count
        else:
            result = msa_runner.query(input_fasta_path)[0]
        with open(msa_out_path, 'w') as f:
            f.write(result[msa_format])
    else:
        logging.warning('Reading MSA from file %s', msa_out_path)
        if msa_format == 'sto' and max_sto_sequences is not None:
            precomputed_msa = parsers.truncate_stockholm_msa(
                msa_out_path, max_sto_sequences)
            result = {'sto': precomputed_msa}
        elif max_sto_sequences is not None:
            with open(msa_out_path, 'r') as f:
                result = {msa_format: "\n".join(f.read().split("\n")[:max_sto_sequences * 2])}
        else:
            with open(msa_out_path, 'r') as f:
                result = {msa_format: f.read()}
    return result


class DataPipeline:
    """Runs the alignment tools and assembles the input features."""

    def __init__(self,
                 jackhmmer_binary_path: str,
                 hhblits_binary_path: str,
                 uniref90_database_path: str,
                 bfd_database_path: Optional[str],
                 uniref30_database_path: Optional[str],
                 small_bfd_database_path: Optional[str],
                 template_searcher: TemplateSearcher,
                 template_featurizer: Optional[templates.TemplateHitFeaturizer],
                 use_small_bfd: bool,
                 # mgnify_max_hits: int = 501, # Removed
                 uniref_max_hits: int = 10000,
                 bfd_max_hits: int = 10000,
                 input_msa: str = None,
                 no_templates: bool = False,
                 use_precomputed_msas: bool = False):
        """Initializes the data pipeline."""
        self._use_small_bfd = use_small_bfd
        self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=uniref90_database_path)

        self.hhblits_uniref30_runner = hhblits.HHBlits(
            binary_path=hhblits_binary_path,
            databases=[uniref30_database_path])

        if use_small_bfd:
            self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=small_bfd_database_path)
        else:
            self.hhblits_bfd_runner = hhblits.HHBlits(
                binary_path=hhblits_binary_path,
                databases=[bfd_database_path])

        self.template_searcher = template_searcher
        self.template_featurizer = template_featurizer
        # self.mgnify_max_hits = mgnify_max_hits # Removed
        self.uniref_max_hits = uniref_max_hits
        self.bfd_max_hits = bfd_max_hits
        self.input_msa = input_msa
        self.no_templates = no_templates
        self.use_precomputed_msas = use_precomputed_msas

    def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
        """Runs alignment tools on the input sequence and creates features."""
        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f'More than one input sequence found in {input_fasta_path}.')
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)

        # 1. Jackhmmer search Uniref90
        uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
        jackhmmer_uniref90_result = run_msa_tool(
            msa_runner=self.jackhmmer_uniref90_runner,
            input_fasta_path=input_fasta_path,
            msa_out_path=uniref90_out_path,
            msa_format='sto',
            use_precomputed_msas=self.use_precomputed_msas,
            max_sto_sequences=self.uniref_max_hits)
        uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])

        # 2. HHblits search UniRef30
        uniref30_out_path = os.path.join(msa_output_dir, 'uniref30_hits.a3m')
        hhblits_uniref30_result = run_msa_tool(
            msa_runner=self.hhblits_uniref30_runner,
            input_fasta_path=input_fasta_path,
            msa_out_path=uniref30_out_path,
            msa_format='a3m',
            use_precomputed_msas=self.use_precomputed_msas,
            max_sto_sequences=self.uniref_max_hits)
        uniref30_msa = parsers.parse_a3m(hhblits_uniref30_result['a3m'])

        # 3. HHblits search BFD (or Jackhmmer for small_bfd)
        if self._use_small_bfd:
            bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
            bfd_result = run_msa_tool(
                msa_runner=self.jackhmmer_small_bfd_runner,
                input_fasta_path=input_fasta_path,
                msa_out_path=bfd_out_path,
                msa_format='sto',
                use_precomputed_msas=self.use_precomputed_msas,
                max_sto_sequences=self.bfd_max_hits)
            bfd_msa = parsers.parse_stockholm(bfd_result['sto'])
        else:
            bfd_out_path = os.path.join(msa_output_dir, 'bfd_hits.a3m')
            bfd_result = run_msa_tool(
                msa_runner=self.hhblits_bfd_runner,
                input_fasta_path=input_fasta_path,
                msa_out_path=bfd_out_path,
                msa_format='a3m',
                use_precomputed_msas=self.use_precomputed_msas,
                max_sto_sequences=self.bfd_max_hits)
            bfd_msa = parsers.parse_a3m(bfd_result['a3m'])

        # mgnify_msa = parsers.Msa(sequences=[], deletion_matrix=[], descriptions=[]) # Removed

        # This will store the final template features dictionary
        final_template_features = {}

        # Conditional template processing
        if self.no_templates:
            logging.info('Skipping template search and featurization as --no_templates is True.')
            # Directly create the dictionary of features expected by the model when templates are skipped.
            # Using STANDARD_RESTYPE_NUM and STANDARD_ATOM_TYPE_NUM defined at the top.
            final_template_features = {
                'template_aatype': np.zeros((0, num_res, STANDARD_RESTYPE_NUM), dtype=np.float32),
                'template_all_atom_positions': np.zeros((0, num_res, STANDARD_ATOM_TYPE_NUM, 3),
                                                     dtype=np.float32),
                'template_sum_probs': np.zeros((0,), dtype=np.float32),
                'template_domain_names': np.array([], dtype=np.object_),
                'template_sequence': np.array([], dtype=np.object_),
                'template_deletion_matrix': np.zeros((0, num_res), dtype=np.float32),
                'template_pseudo_beta': np.zeros((0, num_res, 3), dtype=np.float32),
                'template_pseudo_beta_mask': np.zeros((0, num_res), dtype=np.float32)
            }
        else:
            # Original template search logic
            msa_for_templates = jackhmmer_uniref90_result['sto']
            msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
            msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
                msa_for_templates)

            pdb_hits_out_path = os.path.join(
                msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')

            if not self.use_precomputed_msas or not os.path.isfile(pdb_hits_out_path):
                if self.template_searcher.input_format == 'sto':
                    pdb_templates_result = self.template_searcher.query(msa_for_templates)
                elif self.template_searcher.input_format == 'a3m':
                    uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
                    pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
                else:
                    raise ValueError('Unrecognized template input format: '
                                     f'{self.template_searcher.input_format}')
                with open(pdb_hits_out_path, 'w') as f:
                    f.write(pdb_templates_result)
            else:
                with open(pdb_hits_out_path, 'r') as f:
                    pdb_templates_result = f.read()

            pdb_template_hits = self.template_searcher.get_template_hits(
                output_string=pdb_templates_result, input_sequence=input_sequence)

            if self.template_featurizer is None:
                raise ValueError("Template featurizer is None, but templates are not skipped. "
                                 "Ensure template_featurizer is provided during DataPipeline initialization "
                                 "if --no_templates is False.")

            templates_result = self.template_featurizer.get_templates(
                query_sequence=input_sequence,
                hits=pdb_template_hits)
            final_template_features = templates_result.features


        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res)

        # Changed to only include the MSAs you are actually using
        msa_features = make_msa_features((uniref90_msa, uniref30_msa, bfd_msa))

        logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
        logging.info('Uniref30 MSA size: %d sequences.', len(uniref30_msa))
        logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
        # logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa)) # Removed
        logging.info('Final (deduplicated) MSA size: %d sequences.',
                     msa_features['num_alignments'][0])

        if not self.no_templates:
            logging.info('Total number of templates (NB: this can include bad '
                         'templates and is later filtered to top 4): %d.',
                         final_template_features['template_domain_names'].shape[0])

        return {**sequence_features, **msa_features, **final_template_features}