import sys

for p in ['/home/erschultz', '/home/erschultz/sequences_to_contact_maps']:
    sys.path.insert(1, p)

from sequences_to_contact_maps.result_summary_plots import \
    project_S_to_psi_basis
from sequences_to_contact_maps.utils.argparse_utils import (ArgparserConverter,
                                                            finalize_opt,
                                                            get_base_parser)
from sequences_to_contact_maps.utils.clean_directories import clean_directories
from sequences_to_contact_maps.utils.dataset_classes import make_dataset
from sequences_to_contact_maps.utils.energy_utils import (calculate_E_S,
                                                          calculate_S, s_to_E)
from sequences_to_contact_maps.utils.InteractionConverter import \
    InteractionConverter
from sequences_to_contact_maps.utils.load_utils import (load_all, load_E_S,
                                                        load_final_max_ent_chi,
                                                        load_final_max_ent_S,
                                                        load_X_psi, load_Y,
                                                        load_Y_diag)
from sequences_to_contact_maps.utils.neural_net_utils import (get_dataset,
                                                              load_saved_model)
from sequences_to_contact_maps.utils.plotting_utils import (plot_matrix,
                                                            plot_seq_binary,
                                                            plot_seq_exclusive,
                                                            plot_top_PCs)
from sequences_to_contact_maps.utils.R_pca import R_pca
from sequences_to_contact_maps.utils.utils import (LETTERS, SCC,
                                                   DiagonalPreprocessing,
                                                   calc_dist_strat_corr, crop,
                                                   pearson_round)
from sequences_to_contact_maps.utils.xyz_utils import xyz_load
